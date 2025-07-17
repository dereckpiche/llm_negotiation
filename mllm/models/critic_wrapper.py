import torch, torch.nn as nn, torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

from mllm.models.adapter_wrapper import AdapterWrapper



class ScalarCritic(nn.Module):
    """
    A causal-LM backbone + a scalar value head:
        V_φ(s) = wᵀ h_last + b
    Only LoRA adapters (inside backbone) and the value head are trainable.
    """
    def __init__(self, backbone: AdapterWrapper):
        super().__init__()
        self.backbone = backbone                    
        hidden_size = self.backbone.shared_llm.config.hidden_size
        self.value_head = nn.Linear(hidden_size, 1).to(
            dtype=backbone.dtype, 
            device=backbone.device)

    def forward(self,
                input_ids,
                attention_mask=None,
                **kwargs):
        # AdapterWrapper activates its own adapter internally
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs,
        )
        h_last = outputs.hidden_states[-1]            # (B, S, H)
        values = self.value_head(h_last).squeeze(-1)  # (B, S)
        return values

    def parameters(self, recurse: bool = True):
        """Iterator over *trainable* parameters for this critic."""
        # 1) LoRA params for *this* adapter
        for p in self.backbone.parameters():
            yield p
        # 2) scalar head
        yield from self.value_head.parameters()

    def gradient_checkpointing_enable(self, *args, **kwargs):
        self.backbone.gradient_checkpointing_enable(*args, **kwargs)

    @property
    def dtype(self):
        return self.backbone.dtype

    @property
    def device(self):
        return self.backbone.device