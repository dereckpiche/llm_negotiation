import os
from datetime import datetime

import pandas as pd
import json
import torch
import torch.optim as optim
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from mllm.training.process_training_chat import process_training_chat

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
conv_file = "tests/inputs_for_tests/training_data_convs/conv1.json"


def test_get_assistant_actions_mask_and_score():
    with open(conv_file, "r") as f:
        chat = json.load(f)["chat"]
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    (
        token_ids,
        rewards,
        action_mask, 
        credit_mask,
        state_end_flags
    ) = process_training_chat(
        tokenizer=tokenizer,
        chat_history=chat)

    decoded = tokenizer.convert_ids_to_tokens(token_ids.tolist())
    df = {"Tokens": decoded, "Action Mask": action_mask, "Credit Mask": credit_mask, "State End Flags": state_end_flags}
    df = pd.DataFrame(df)
    print(df.to_string())
    print(rewards)
   

if __name__ == "__main__":
    test_get_assistant_actions_mask_and_score()
