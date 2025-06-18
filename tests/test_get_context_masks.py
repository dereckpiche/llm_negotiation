import os
from datetime import datetime

import pandas as pd
import torch
import torch.optim as optim
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from training.get_context_masks import *

model_name = "Qwen/Qwen2.5-0.5B-Instruct"


def test_get_assistant_actions_mask_and_score():
    conv = [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "The dog is hollow."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "The cat is blue."},
    ]
    per_message_score = torch.Tensor([0.123, 0.234])
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    token_ids = tokenizer.apply_chat_template(
        conv, 
        return_tensors="pt"
    )
    print(tokenizer.eos_token_id)

    (
        action_mask, 
        action_timestamps,
        state_end_flags
    ) = get_context_masks(
        tokenizer=tokenizer,
        token_ids=token_ids)


    decoded = tokenizer.convert_ids_to_tokens(token_ids.tolist()[0])
    df = {"Tokens": decoded, "Action Mask": action_mask, "Action Timestamps": action_timestamps, "Action End Flags": state_end_flags}
    df = pd.DataFrame(df)
    print(df)
   

if __name__ == "__main__":
    test_get_assistant_actions_mask_and_score()
