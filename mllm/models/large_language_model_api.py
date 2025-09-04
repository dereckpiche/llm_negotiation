from __future__ import annotations

import asyncio
import copy
import os
import re
from typing import Any, Callable, Dict, List, Optional, Sequence

from openai import AsyncOpenAI

from mllm.models.inference_backend import PolicyOutput


class LargeLanguageModelOpenAI:
    """Tiny async wrapper for OpenAI Chat Completions."""

    def __init__(
        self,
        use_reasoning: bool,
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
        self.use_reasoning = use_reasoning

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
        if use_reasoning:
            self.sampling_params["reasoning"] = {
                "effort": "medium",
                "summary": "detailed",
            }
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

    def extract_output_from_response(self, resp: Response) -> PolicyOutput:
        if self.use_reasoning:
            summary = resp.output[0].summary
            if summary != []:
                reasoning_content = summary[0].text
                reasoning_content = f"OpenAI Reasoning Summary: {reasoning_content}"
            else:
                reasoning_content = None
            content = resp.output[1].content[0].text

        else:
            reasoning_content = None
            content = resp.output[0].content[0].text

        return PolicyOutput(
            content=content,
            reasoning_content=reasoning_content,
        )

    async def generate(
        self,
        prompt: list[dict],
        regex: Optional[str] = None,
    ) -> PolicyOutput:
        # Remove any non-role/content keys from the prompt else openai will error
        prompt = [{"role": p["role"], "content": p["content"]} for p in prompt]

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
                resp = await self.client.responses.create(
                    model=self.model,
                    input=prompt,
                    **self.sampling_params,
                )
                policy_output = self.extract_output_from_response(resp)
                if pattern.fullmatch(policy_output.content):
                    return policy_output
                prompt = [
                    *prompt,
                    {
                        "role": "user",
                        "content": (
                            f"Invalid response format. Expected format (regex): {regex}\n Please try again and provide ONLY a response that matches this regex."
                        ),
                    },
                ]
            return policy_output

        # Simple, unconstrained generation
        resp = await self.client.responses.create(
            model=self.model,
            input=prompt,
            **self.sampling_params,
        )
        policy_output = self.extract_output_from_response(resp)
        return policy_output

    def shutdown(self) -> None:
        self.client = None
