"""
To debug the code without loading LLM engines.
"""

from typing import List

from mllm.models.local_llm import *
from mllm.utils.common_imports import *
from mllm.models.lean_local_llm import LeanLocalLLM
from collections.abc import Callable

class DummyLocalLLM(LeanLocalLLM):
    def __init__(self, *args, **kwargs):
        self.adapter_ids = list(kwargs["adapter_configs"].keys())

    def toggle_training_mode(self):
        pass

    def toggle_eval_mode(self):
        pass

    def set_adapter_eval(self, adapter_id: str) -> None:
        pass

    def get_trainable_objects(self) -> dict:
        return {}

    def log_gpu_usage(self, message: str) -> None:
        pass

    def get_callable_objects(self) -> dict[str, Callable]:
        policies = {id:lambda x : self.generate(x) for id in self.adapter_ids}
        return policies

    def export_current_adapter(self) -> None:
        pass

    def generate(self, prompt):
        return "C"
