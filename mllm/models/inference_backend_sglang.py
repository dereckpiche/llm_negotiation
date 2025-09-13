# new_backend_sglang_offline.py
from __future__ import annotations

import asyncio
from typing import Any, Optional

# import sglang as sgl

from mllm.models.inference_backend import LLMInferenceBackend


class SGLangOfflineBackend(LLMInferenceBackend):
    def __init__(
        self,
        model_name: str,
        tokenizer,  # unused but kept for parity
        adapter_paths: dict[str, str],
        device: str = "cuda",
        max_model_len: Optional[int] = None,
        enable_lora: bool = True,
        lora_target_modules: Optional[list[str] | str] = None,
        max_loras_per_batch: int = 8,
        engine_kwargs: dict[str, Any] = None,
    ):
        self.model_name = model_name
        self.adapter_paths = adapter_paths
        self.current_adapter: Optional[str] = None
        engine_kwargs = dict(engine_kwargs or {})
        # Map server-style LoRA flags to offline engine ctor
        if enable_lora and adapter_paths:
            engine_kwargs.setdefault("enable_lora", True)
            # The offline Engine mirrors server args; pass a mapping name->path
            engine_kwargs.setdefault("lora_paths", adapter_paths)
            if lora_target_modules is not None:
                engine_kwargs.setdefault("lora_target_modules", lora_target_modules)
            engine_kwargs.setdefault("max_loras_per_batch", max_loras_per_batch)

        if max_model_len is not None:
            engine_kwargs.setdefault("context_length", max_model_len)

        # Launch in-process engine (no HTTP server)
        self.llm = sgl.Engine(model_path=model_name, **engine_kwargs)  # async-ready
        # SGLang supports: generate(), async_generate(), and async streaming helpers. :contentReference[oaicite:2]{index=2}

    def is_ready(self) -> bool:
        return True

    def toggle_training_mode(self) -> None:
        # No explicit KV release API offline; typically you pause usage here.
        pass

    def toggle_eval_mode(self) -> None:
        pass

    def shutdown(self) -> None:
        # Engine cleans up on GC; explicit close not required.
        pass

    def prepare_adapter(self, adapter_id: Optional[str]) -> None:
        # With offline Engine, when LoRA is enabled at init,
        # you select adapter per request via the input batch mapping.
        self.current_adapter = adapter_id

    async def generate(
        self, prompt_text: str, sampling_params: dict, adapter_id: Optional[str]
    ) -> str:
        # Non-streaming async (batch of 1). For batched prompts, pass a list.
        params = {
            "temperature": sampling_params.get("temperature", 1.0),
            "top_p": sampling_params.get("top_p", 1.0),
            "max_new_tokens": sampling_params.get("max_new_tokens", 128),
        }
        if (tk := sampling_params.get("top_k", -1)) and tk > 0:
            params["top_k"] = tk
        if (mn := sampling_params.get("min_new_tokens")) is not None:
            params["min_new_tokens"] = mn
        if (fp := sampling_params.get("frequency_penalty")) is not None:
            params["frequency_penalty"] = fp

        # If using multi-LoRA, SGLang lets you provide adapter names aligned to each input.
        prompts = [prompt_text]
        adapters = [adapter_id] if adapter_id else None  # or omit for base
        outs = await self.llm.async_generate(
            prompts, params, adapters
        )  # :contentReference[oaicite:3]{index=3}
        return outs[0]["text"]
