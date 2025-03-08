from typing import List
from utils.common_imports import *

from models.local_llm import *

from typing import List
from utils.common_imports import *

from models.local_llm import *

class DummyLocalLLM(LocalLLM):
    def __init__(self, *args, **kwargs):
        pass

    def prepare_adapter_train(self, adapter_name: str):
        pass

    def prepare_adapter_eval(self, adapter_name: str, seed_offset: int = 0):
        pass

    def destroy_hf(self):
        pass

    def destroy_vllm(self):
        pass

    def log_gpu_usage(self, message: str) -> None:
        pass

    def prompt(self, contexts) -> str:
        return [""] * len(contexts)

    def export_current_adapter(self) -> None:
        pass

    def use_hf_model(self):
        pass

    def use_vllm_model(self):
        pass

    def set_adapter(self, adapter_name):
        pass

