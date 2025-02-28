from typing import List
from utils.common_imports import *

from llms.local_llm import *

class Dummy (LocalLLM):
    def prompt(self, contexts) -> str:
        return ["" for item in contexts]
    def use_hf_model(self): return
    def use_vllm_model(self): return
    def set_adapter(self, adapter_name): return
    def train_ppo(
            self, queries: List, responses: List, scores: List[float]
        ) -> dict: return
