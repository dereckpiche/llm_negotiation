from utils.common_imports import *

import torch
import os
import json

def get_conversations(folder_path: str):
    conversations = []
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        with open(file_path, 'r') as f:
            conversations.append(json.load(f))
    return conversations

def conversation_to_rl_data(tokenizer,
                            conversation,
                            ):

    # Check if the tokenizer has an EOS token
    if tokenizer.eos_token is None:
        raise ValueError("The tokenizer does not have an EOS token.")

    # Apply chat template to the entire conversation, include the last assistant message
    formatted_conversation = tokenizer.apply_chat_template(
        conversation, add_generation_prompt=False, tokenize=False, use_system_prompt=False
    )
    tokens = tokenizer.encode(
        formatted_conversation, return_tensors='pt', add_special_tokens=False
    ).squeeze(0)

    # Find all <|eot_id|> token positions (TODO: Handle tokenizers without eos_token_id)
    eot_id = tokenizer.eos_token_id
    all_eot_positions = (tokens == eot_id).nonzero(as_tuple=True)[0].tolist()

    # Remove the first <|eot_id|> position which corresponds to the system prompt
    eot_positions = all_eot_positions[1:]

    score_values = []
    output_mask = []
    current_position = 0

    # Associate return values and output masks based on adjusted <|eot_id|> positions
    for i, message in enumerate(conversation):
        score_value = message.get('score', 0)  # Default to 0 if 'score' is not present. This line is not necessary I think.

        if i < len(eot_positions):
            next_position = eot_positions[i] + 1  # Include the <|eot_id|> token
        else:
            next_position = len(tokens)

        if message.get('role', None) == "assistant":
            mask_value = 1 
            # TODO: for Llama <|start_header_id|>assistant<|end_header_id|>\n\n takes 4 tokens, where mask is still zero
            segment_length = next_position - current_position
            score_values.extend([0] * 4 + [score_value] * (segment_length - 4))
            output_mask.extend([0] * 4 + [mask_value] * (segment_length - 4))
        else:
            mask_value = 0 # only train on messages from the assistant
            # Extend return values and output masks for the current segment
            segment_length = next_position - current_position
            score_values.extend([score_value] * segment_length)
            output_mask.extend([mask_value] * segment_length)
        current_position = next_position

    return_tensor = torch.tensor(score_values)
    output_mask_tensor = torch.tensor(output_mask)
    # Commented below since add_generation_prompt=False in tokenizer.encode
    # last_eot_index = (tokens == eot_id).nonzero(as_tuple=True)[0][-1].item()
    # tokens = tokens[:last_eot_index+1]
    # output_mask_tensor = output_mask_tensor[:last_eot_index+1]

    return tokens, return_tensor, output_mask_tensor

def conversations_to_rl_data(tokenizer, conversations):
    contexts = []
    scores = []
    output_masks = []

    for conversation in conversations:
        if conversation:
            context_tensor, return_tensor, output_mask_tensor = conversation_to_rl_data(tokenizer, conversation)
            contexts.append(context_tensor)
            scores.append(return_tensor)
            output_masks.append(output_mask_tensor)

    return contexts, scores, output_masks

def paths_to_rl_data(tokenizer, paths):
    conversations = []
    for path in paths:
        conversations.extend(get_conversations(path))
    return conversations_to_rl_data(tokenizer, conversations)