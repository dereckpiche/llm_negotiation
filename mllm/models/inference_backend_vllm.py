import asyncio
from typing import Optional

from transformers import AutoTokenizer
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.sampling_params import GuidedDecodingParams, RequestOutputKind

from mllm.models.inference_backend import LLMInferenceBackend
from mllm.utils.short_id_gen import generate_short_id


class VLLMAsyncBackend(LLMInferenceBackend):
    def __init__(
        self,
        model_name: str,
        tokenizer: AutoTokenizer,
        adapter_paths: dict[str, str],
        engine_init_kwargs: dict = {},
        sampling_params: dict = {},
    ):
        self.model_name = model_name
        self.adapter_paths = adapter_paths or {}
        self.current_adapter = None
        self.vllm_adapter_ids = {
            adapter_id: generate_short_id() for adapter_id in adapter_paths.keys()
        }
        ea = dict(model=model_name, **engine_init_kwargs)
        ea["enable_lora"] = True
        ea["max_loras"] = len(self.vllm_adapter_ids)
        ea["enable_sleep_mode"] = True
        self.engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**ea))

        self.sampling_params = sampling_params

    def prepare_adapter(
        self, adapter_id: Optional[str], weights_got_updated: bool
    ) -> None:
        self.current_adapter = adapter_id
        if weights_got_updated:
            self.vllm_adapter_ids[adapter_id] = generate_short_id()
        self.current_lora_request = LoRARequest(
            adapter_id,
            self.vllm_adapter_ids[adapter_id],
            self.adapter_paths[adapter_id],
        )

    async def toggle_training_mode(self) -> None:
        await self.engine.sleep(level=1)

    async def toggle_eval_mode(self) -> None:
        await self.engine.wake_up()

    def shutdown(self) -> None:
        # No explicit close call; engine stops when process exits.
        pass

    async def generate(self, prompt_text: str, regex: Optional[str] = None) -> str:
        # Build SamplingParams correctly

        guided = GuidedDecodingParams(regex=regex) if regex else None
        sp = SamplingParams(
            **self.sampling_params,
            guided_decoding=guided,
            output_kind=RequestOutputKind.FINAL_ONLY,
        )

        request_id = f"req-{asyncio.get_running_loop().time()}"
        results = self.engine.generate(
            prompt_text,
            sp,  # SamplingParams(...)
            request_id,
            lora_request=self.current_lora_request,
        )

        async for out in results:  # with FINAL_ONLY this runs once
            res = out.outputs[0].text
        return res
