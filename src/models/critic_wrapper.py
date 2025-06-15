import torch, torch.nn as nn, torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model


class ScalarCritic(nn.Module):
    """
    A causal-LM backbone + a scalar value head:
        V_φ(s) = wᵀ h_last + b
    Only LoRA adapters (inside backbone) and the value head are trainable.
    """
    def __init__(self, backbone: AutoModelForCausalLM):
        super().__init__()
        self.backbone = backbone
        hidden_size   = backbone.config.hidden_size
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        **kwargs,
    ):
        # Need hidden states, so set output_hidden_states=True
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs,
        )
        hidden_states = outputs.hidden_states[-1]     # shape (B, S, H)
        values  = self.value_head(hidden_states).squeeze(-1)       # shape (B, S)
        return values