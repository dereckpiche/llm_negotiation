import asyncio
from typing import Optional

import rstr
from transformers import AutoTokenizer

from mllm.models.inference_backend import LLMInferenceBackend
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

    def toggle_training_mode(self) -> None:
        pass

    def toggle_eval_mode(self) -> None:
        pass

    def shutdown(self) -> None:
        pass

    async def generate(self, prompt_text: str, regex: Optional[str] = None) -> str:
        if regex:
            # Create random string that respects the regex
            return rstr.xeger(regex)
        else:
            return "I am a dummy backend without a regex."
