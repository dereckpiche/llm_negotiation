import asyncio
import re
from typing import Optional

import rstr
from transformers import AutoTokenizer

from mllm.models.inference_backend import LLMInferenceBackend, PolicyOutput
from mllm.utils.short_id_gen import generate_short_id


class DummyInferenceBackend(LLMInferenceBackend):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        pass

    def prepare_adapter(
        self, adapter_id: Optional[str], weights_got_updated: bool
    ) -> None:
        pass

    async def toggle_training_mode(self) -> None:
        await asyncio.sleep(0)
        pass

    async def toggle_eval_mode(self) -> None:
        await asyncio.sleep(0)
        pass

    def shutdown(self) -> None:
        pass

    async def generate(
        self, prompt_text: str, regex: Optional[str] = None
    ) -> PolicyOutput:
        content = "I am a dummy backend without a regex."
        reasoning_content = None

        if regex:
            raw_text = rstr.xeger(regex)
            content = raw_text
            # Strict split: require \n<think>...</think>\n\n before final content
            m = re.match(
                r"^\n<think>\n([\s\S]*?)</think>\n\n(.*)$", raw_text, flags=re.DOTALL
            )
            if m:
                reasoning_content = m.group(1)
                content = m.group(2)

        return PolicyOutput(content=content, reasoning_content=reasoning_content)
