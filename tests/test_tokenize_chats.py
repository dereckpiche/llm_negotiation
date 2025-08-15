import os
from datetime import datetime
import pandas as pd
import torch
import torch.optim as optim
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from mllm.training.tokenize_chats import *
from mllm.training.training_data_utils import TrainingChatTurn
from mllm.training.training_data_utils import get_causal_reasoning_mask

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

training_chat = [
            TrainingChatTurn(time_step=0, role="user", agent_id="human", content="Is the local group co-operative?", reasoning_content=None, is_state_end=True),
    TrainingChatTurn(time_step=0, role="assistant", agent_id="model", content="The dog is hollow.", reasoning_content="Very deep thoughts.", is_state_end=False),
    TrainingChatTurn(time_step=1, role="user", agent_id="human", content="Howard Hughes was a very rich man.", reasoning_content=None, is_state_end=True),
    TrainingChatTurn(time_step=1, role="assistant", agent_id="model", content="And a talented pilot.", reasoning_content=None, is_state_end=False),
]


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    input_ids, action_mask, timesteps, state_ends_mask, reasoning_limit_tuples = process_training_chat(tokenizer, training_chat)
    

    reasoning_token_mask = torch.zeros(input_ids.numel(), dtype=torch.bool)# TODO (S,)
    print(reasoning_limit_tuples)
    for l in reasoning_limit_tuples: reasoning_token_mask[l[0]:l[1]] = True

    reasoning_token_attention_mask = get_causal_reasoning_mask((1, input_ids.numel()), [reasoning_limit_tuples])[0] # TODO (S,S)
    attends_to = [tokenizer.decode(input_ids[tmask==True]) for tmask in reasoning_token_attention_mask]
    df = {
        'Tokens': tokenizer.convert_ids_to_tokens(input_ids.tolist()), 
        'Action Mask': action_mask, 
        'Timesteps': timesteps, 
        'State Ends Mask': state_ends_mask, 
        'Reasoning Token Mask': reasoning_token_mask, 
        'Attention Attends To': attends_to,
    }
    df = pd.DataFrame(df)
    print(df)
    output_path = "tests/outputs_for_tests/test_tokenize_chats.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
