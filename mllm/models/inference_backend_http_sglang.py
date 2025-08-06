import os

import httpx
import requests
from sglang.utils import launch_server_cmd, wait_for_server

from mllm.models.inference_backend import LLMInferenceBackend


class HttpSGLangBackend(LLMInferenceBackend):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.port = None
        self.proc = None
        self.urls = {}
        # track sglang adapter ids separately from your logical ids
        self.sglang_names = {aid: aid for aid in self.adapter_paths.keys()}
        self.needs_loading = {aid: True for aid in self.adapter_paths.keys()}

        # defaults you already used:
        self.mem_fraction = kwargs.get("mem_fraction_static", 0.6)
        self.dtype = kwargs.get("dtype", "bfloat16")
        self.extra_cli = kwargs.get("extra_cli", "")
        self.disable_radix_cache = kwargs.get("disable_radix_cache", True)

    def launch(self) -> None:
        # find local hf cache path for server
        from transformers.utils import cached_file

        local_llm_path = os.path.split(cached_file(self.model_name, "config.json"))[0]

        lora_str = ""
        if self.adapter_paths:
            lora_str = "--lora-paths " + " ".join(
                f"{aid}={path}" for aid, path in self.adapter_paths.items()
            )

        cmd = f"""
        python3 -m sglang.launch_server --model-path {local_llm_path} \
        --host 0.0.0.0 {lora_str} \
        {'--disable-radix-cache' if self.disable_radix_cache else ''} \
        --mem-fraction-static {self.mem_fraction} --dtype {self.dtype} {self.extra_cli}
        """
        self.proc, self.port = launch_server_cmd(cmd)
        wait_for_server(f"http://localhost:{self.port}")
        base = f"http://localhost:{self.port}"
        self.urls = dict(
            generate=f"{base}/generate",
            release=f"{base}/release_memory_occupation",
            resume=f"{base}/resume_memory_occupation",
            load_lora=f"{base}/load_lora_adapter",
            unload_lora=f"{base}/unload_lora_adapter",
        )

    def is_ready(self) -> bool:
        try:
            requests.get(self.urls["generate"], timeout=2)
            return True
        except Exception:
            return False

    def prepare_adapter(self, adapter_id: str) -> None:
        if adapter_id is None:
            return
        if self.needs_loading.get(adapter_id, False):
            # unload old name if present
            try:
                requests.post(
                    self.urls["unload_lora"],
                    json={"lora_name": self.sglang_names[adapter_id]},
                    timeout=10,
                )
            except Exception:
                pass
            new_name = self._short_id()
            self.sglang_names[adapter_id] = new_name
            requests.post(
                self.urls["load_lora"],
                json={
                    "lora_name": new_name,
                    "lora_path": self.adapter_paths[adapter_id],
                },
            ).raise_for_status()
            self.needs_loading[adapter_id] = False

    async def generate(
        self, prompt_text: str, sampling_params: dict, adapter_id: str | None
    ) -> str:
        lora_name = self.sglang_names.get(adapter_id) if adapter_id else None
        payload = {
            "text": [prompt_text],
            "sampling_params": sampling_params,
        }
        if lora_name:
            payload["lora_path"] = [lora_name]

        timeout = httpx.Timeout(3600.0, connect=3600.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(self.urls["generate"], json=payload)
            resp.raise_for_status()
            return resp.json()[0]["text"]

    def toggle_training_mode(self) -> None:
        # free KV space while training adapters
        requests.post(
            self.urls["release"], json={"tags": ["kv_cache"]}
        ).raise_for_status()

    def toggle_eval_mode(self) -> None:
        # re-allocate KV space
        try:
            requests.post(
                self.urls["resume"], json={"tags": ["kv_cache"]}
            ).raise_for_status()
        except Exception:
            pass

    def shutdown(self) -> None:
        from sglang.utils import terminate_process

        if self.proc:
            terminate_process(self.proc)

    def _short_id(self) -> str:
        import uuid

        return str(uuid.uuid4().int)[:8]
