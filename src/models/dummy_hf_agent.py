from typing import Any
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from trl import SFTTrainer
from peft import get_peft_model, LoraConfig, TaskType
import os

from models.hf_agent import *

class DummyHfAgent(HfAgent):
    def prompt(self, contexts) -> str: 
        return ["" for item in contexts]
    def prepare_adapter_train(self, adapter_name: str): return
    def prepare_adapter_eval(self, adapter_name: str): return 
    def destroy_hf(self): return 
    def destroy_vllm(self): return 
    def export_current_adapter(self): return 