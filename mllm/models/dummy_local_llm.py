"""
To debug the code without loading LLM engines.
"""

from typing import List

from mllm.models.lean_local_llm import LeanLocalLLM
from collections.abc import Callable

class DummyLocalLLM(LeanLocalLLM):
    # def __init__(self, *args, **kwargs):
    #     self.name = kwargs["name"]
    #     self.adapter_ids = list(kwargs["adapter_configs"].keys())
    #
    def init_sg_lang_server(self):
        """
        Here we don't want to use the inference engine.
        """
        pass

    def toggle_training_mode(self):
        pass

    def toggle_eval_mode(self):
        pass

    def log_gpu_usage(self, message: str) -> None:
        pass

    # def get_inference_policies(self) -> dict[str, Callable]:

    #     policies = {self.name+"/"+id: async lambda x : await self.generate(x) for id in self.adapter_ids}
    #     return policies
    #
    def prepare_adapter_for_inference(self, adapter_id: str) -> None:
        pass

    async def generate(self, prompt):
        import random
        n: float = random.random()
        if n < 0.5: return "C"
        return "D"
