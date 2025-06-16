
import torch.nn as nn
from typing import Iterable, Tuple
import torch
from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)

class AdapterWrapper(nn.Module):
    """
    A thin façade that
      • keeps a reference to a *shared* PEFT-wrapped model,
      • ensures `set_adapter(adapter)` is called on every forward,
      • exposes only the parameters that should be trained for that adapter
        (plus whatever extra modules you name).
    """
    def __init__(
        self,
        shared_llm: nn.Module,
        adapter_id: str,
        lora_config: dict
        ):
        super().__init__()
        self.shared_llm = shared_llm
        self.shared_llm.train()
        self.adapter_id = adapter_id
        lora_config = LoraConfig(**lora_config)
        # this modifies the shared llm in place, adding a lora adapter inside
        self.shared_llm = get_peft_model(
            model=shared_llm,
            peft_config=lora_config,
            adapter_name=adapter_id,
        )

    def parameters(self, recurse: bool = True):
        """
        "recurse" is just for pytorch compatibility
        """
        self.shared_llm.set_adapter(self.adapter_id)
        params = [p for p in self.shared_llm.parameters() if p.requires_grad]

        return params
   
    def forward(self, *args, **kwargs):
        self.shared_llm.set_adapter(self.adapter_id)
        return self.shared_llm(*args, **kwargs)
    
    def save_pretrained(self, save_path):
        self.shared_llm.save_pretrained(save_path)

    def gradient_checkpointing_enable(self, *args, **kwargs):
        self.shared_llm.gradient_checkpointing_enable(*args, **kwargs)
    