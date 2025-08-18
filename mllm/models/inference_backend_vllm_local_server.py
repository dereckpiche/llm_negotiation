import json
import os
import subprocess
import time

import httpx
import requests

from mllm.models.inference_backend import LLMInferenceBackend


class HttpVLLMBackend(LLMInferenceBackend):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.port = kwargs.get("port", 8000)
        self.host = kwargs.get("host", "0.0.0.0")
        self.proc = None
        self.base_url = f"http://{self.host}:{self.port}"
        # vLLM memory safety knobs
        self.gpu_mem_util = kwargs.get("gpu_memory_utilization", 0.9)
        self.max_model_len = kwargs.get("max_model_len", None)
        self.max_num_seqs = kwargs.get("max_num_seqs", None)
        self.max_batched_tokens = kwargs.get("max_num_batched_tokens", None)
        self.dtype = kwargs.get("dtype", "bfloat16")
        self.trust_remote_code = kwargs.get("trust_remote_code", False)
        # LoRA strategy: "preload" (CLI) or "runtime" (endpoints) depending on your vLLM build
        self.lora_mode = kwargs.get(
            "lora_mode", "preload"
        )  # "runtime" supported in newer builds
        self.runtime_lora_enabled = self.lora_mode == "runtime"

        # If preloading: build CLI args (adapter name -> path)
        self._preload_lora_args = []
        if self.adapter_paths and self.lora_mode == "preload":
            # vLLM supports multiple LoRA modules via CLI in recent versions
            # Example flag shapes can vary; adapt as needed for your version:
            # --lora-modules adapter_id=path
            for aid, pth in self.adapter_paths.items():
                self._preload_lora_args += ["--lora-modules", f"{aid}={pth}"]

    def launch(self):
        # Build vLLM serve command
        cmd = [
            "python3",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            self.model_name,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--dtype",
            self.dtype,
            "--gpu-memory-utilization",
            str(self.gpu_mem_util),
        ]
        if self.trust_remote_code:
            cmd += ["--trust-remote-code"]
        if self.max_model_len:
            cmd += ["--max-model-len", str(self.max_model_len)]
        if self.max_num_seqs:
            cmd += ["--max-num-seqs", str(self.max_num_seqs)]
        if self.max_batched_tokens:
            cmd += ["--max-num-batched-tokens", str(self.max_batched_tokens)]
        cmd += self._preload_lora_args

        self.proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        self._wait_ready()

    def _wait_ready(self, timeout=120):
        url = f"{self.base_url}/v1/models"
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                r = requests.get(url, timeout=2)
                if r.status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(1)
        raise RuntimeError("vLLM server did not become ready in time")

    def is_ready(self) -> bool:
        try:
            return (
                requests.get(f"{self.base_url}/v1/models", timeout=2).status_code == 200
            )
        except Exception:
            return False

    def prepare_adapter(self, adapter_id: str) -> None:
        if not adapter_id or not self.runtime_lora_enabled:
            return
        # Newer vLLM builds expose runtime LoRA endpoints. If yours differs,
        # adjust the path/body here and keep the interface stable.
        try:
            requests.post(
                f"{self.base_url}/v1/load_lora_adapter",
                json={
                    "adapter_name": adapter_id,
                    "adapter_path": self.adapter_paths[adapter_id],
                },
                timeout=10,
            ).raise_for_status()
        except Exception as e:
            # If already loaded or endpoint not present, swallow or log
            pass

    async def generate(
        self, prompt_text: str, sampling_params: dict, adapter_id: str | None
    ) -> str:
        # Map your sampling params to OpenAI schema
        body = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt_text}],
            "temperature": sampling_params.get("temperature", 1.0),
            "top_p": sampling_params.get("top_p", 1.0),
            "max_tokens": sampling_params.get("max_new_tokens", 128),
        }
        # Optional knobs:
        if sampling_params.get("top_k", -1) and sampling_params["top_k"] > 0:
            # vLLM accepts top_k via extra params; put under "extra_body"
            body.setdefault("extra_body", {})["top_k"] = sampling_params["top_k"]
        if sampling_params.get("min_new_tokens", None) is not None:
            body.setdefault("extra_body", {})["min_tokens"] = sampling_params[
                "min_new_tokens"
            ]
        if sampling_params.get("frequency_penalty", None) is not None:
            body["frequency_penalty"] = sampling_params["frequency_penalty"]

        # Select LoRA adapter
        if adapter_id:
            if self.runtime_lora_enabled:
                body.setdefault("extra_body", {})["lora_adapter"] = adapter_id
            else:
                # when preloaded via CLI, most builds select by name via "adapter_name"/"lora_adapter"
                body.setdefault("extra_body", {})["lora_adapter"] = adapter_id

        url = f"{self.base_url}/v1/chat/completions"
        timeout = httpx.Timeout(3600.0, connect=3600.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=body)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    def toggle_training_mode(self) -> None:
        # vLLM doesn’t expose an explicit KV “release” toggle via API.
        # Strategy: keep inference server idle during training, or run training in a separate process.
        pass

    def toggle_eval_mode(self) -> None:
        pass

    def shutdown(self) -> None:
        if self.proc:
            self.proc.terminate()
