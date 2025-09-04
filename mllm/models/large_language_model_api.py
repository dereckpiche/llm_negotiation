from __future__ import annotations

import asyncio
import os
import re
from typing import Any, Callable, Dict, List, Optional, Sequence

from openai import AsyncOpenAI

from mllm.models.inference_backend import PolicyOutput


class LargeLanguageModelOpenAI:
    """Tiny async wrapper for OpenAI Chat Completions."""

    def __init__(
        self,
        llm_id: str,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout_s: float = 300.0,
        regex_max_attempts: int = 10,
        sampling_params: Optional[Dict[str, Any]] = None,
        output_directory: Optional[str] = None,
    ) -> None:
        self.llm_id = llm_id
        self.model = model
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "Set OPENAI_API_KEY as global environment variable or pass api_key."
            )
        client_kwargs: Dict[str, Any] = {"api_key": key, "timeout": timeout_s}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = AsyncOpenAI(**client_kwargs)

        # Sampling/default request params set at init
        self.sampling_params = sampling_params
        self.regex_max_attempts = max(1, int(regex_max_attempts))

    def get_inference_policies(self) -> Dict[str, Callable]:
        return {
            self.llm_id: self.generate,
        }

    async def prepare_adapter_for_inference(self, *args: Any, **kwargs: Any) -> None:
        await asyncio.sleep(0)
        pass

    async def toggle_eval_mode(self, *args: Any, **kwargs: Any) -> None:
        await asyncio.sleep(0)
        pass

    async def toggle_training_mode(self, *args: Any, **kwargs: Any) -> None:
        await asyncio.sleep(0)
        pass

    async def export_adapters(self, *args: Any, **kwargs: Any) -> None:
        await asyncio.sleep(0)
        pass

    async def checkpoint_all_adapters(self, *args: Any, **kwargs: Any) -> None:
        await asyncio.sleep(0)
        pass

    async def generate(
        self,
        prompt: list[dict],
        regex: Optional[str] = None,
    ) -> PolicyOutput:
        # If regex is required, prime the model and validate client-side
        if regex:
            constraint_msg = {
                "role": "user",
                "content": (
                    f"Output must match this regex exactly: {regex} \n"
                    "Return only the matching string, with no quotes or extra text."
                ),
            }
            prompt = [constraint_msg, *prompt]
            pattern = re.compile(regex)
            for _ in range(self.regex_max_attempts):
                resp = await self.client.chat.completions.create(
                    model=self.model,
                    messages=prompt,
                    **self.sampling_params,
                )
                text = (resp.choices[0].message.content or "").strip()
                nb_reasoning_tokens = (
                    resp.usage.completion_tokens_details.reasoning_tokens
                )
                if pattern.fullmatch(text):
                    return PolicyOutput(
                        content=text,
                        reasoning_content=f"{nb_reasoning_tokens} reasoning tokens hidden by OpenAI.",
                    )
                prompt = [
                    *prompt,
                    {
                        "role": "user",
                        "content": (
                            f"Invalid response format. Expected format (regex): {regex}\n Please try again and provide ONLY a response that matches this regex."
                        ),
                    },
                ]
            return PolicyOutput(
                content=text,
                reasoning_content=f"{nb_reasoning_tokens} reasoning tokens hidden by OpenAI.",
            )

        # Simple, unconstrained generation
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=prompt,
            **self.sampling_params,
        )
        return PolicyOutput(
            content=resp.choices[0].message.content,
            reasoning_content=f"{resp.usage.completion_tokens_details.reasoning_tokens} reasoning tokens hidden by OpenAI.",
        )

    def shutdown(self) -> None:
        self.client = None
