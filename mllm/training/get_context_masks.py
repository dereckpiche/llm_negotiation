from transformers import AutoTokenizer
import numpy as np
import torch




def get_sentencepieced_example(tokenizer: AutoTokenizer):
    conv_example = [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    ]
    token_ids = tokenizer.apply_chat_template(
        conv_example, 
        add_generation_prompt=False,
        use_system_prompt=False,
        return_tensors="pt").tolist()[0]
    sentencepieces = tokenizer.convert_ids_to_tokens(token_ids)
    sps = []
    for tid, sp in zip(token_ids, sentencepieces):
        sps.append((tid, sp))
    return sps


def get_context_masks(
    tokenizer: AutoTokenizer,
    token_ids:torch.Tensor):
    """
    TODO: docstring
    Args:
        assistant_msg_scores:
            Score attributed to each assistant messages. Length is same
            as number of assistant messages in conversation. 
    Returns:
         action_timestamps: 
           action_timestamps[i] = t means that token_ids[i] belongs to the t'th action.
           (Each response of the model is considered an action.)
           action_timestamps[i] = -1 means that it was not part of an action. (Part of user message.)
    """

    tokenizer_name = tokenizer.name_or_path
    token_ids = token_ids.squeeze()
    nb_tokens = token_ids.shape[0]
    action_mask = torch.zeros(size=token_ids.shape)
    action_timestamps = np.full(shape=token_ids.shape, fill_value=-1.0)
    state_end_flags = np.full(shape=token_ids.shape, fill_value=False)


    if tokenizer_name in ["Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"]:
        """
        For this tokenizer, eos_token_id is 151645 and get_sentencepieced_example(qwen_tokenizer) returns

            [(151644, '<|im_start|>'), (8948, 'system'), (198, 'Ċ'), (2610, 'You'), (525, 'Ġare'), (1207, 'ĠQ'), (16948, 'wen'), (11, ','), (3465, 'Ġcreated'), (553, 'Ġby'), (54364, 'ĠAlibaba'), (14817, 'ĠCloud'), (13, '.'), (1446, 'ĠYou'), (525, 'Ġare'), (264, 'Ġa'), (10950, 'Ġhelpful'), (17847, 'Ġassistant'), (13, '.'), (151645, '<|im_end|>'), (198, 'Ċ'), (151644, '<|im_start|>'), (872, 'user'), (198, 'Ċ'), (1112, '...'), (151645, '<|im_end|>'), (198, 'Ċ'), (151644, '<|im_start|>'), (77091, 'assistant'), (198, 'Ċ'), (1112, '...'), (151645, '<|im_end|>'), (198, 'Ċ'), (151644, '<|im_start|>'), (872, 'user'), (198, 'Ċ'), (1112, '...'), (151645, '<|im_end|>'), (198, 'Ċ'), (151644, '<|im_start|>'), (77091, 'assistant'), (198, 'Ċ'), (1112, '...'), (151645, '<|im_end|>'), (198, 'Ċ')]
        """
        am_count = -1 # assistant message count
        assistant_turn = False
        pointer = 0
        while pointer < nb_tokens:
            if token_ids[pointer] == 151644 and token_ids[pointer+1] == 77091:
                pointer += 3
                assistant_turn = True
                state_end_flags[pointer] = True
                am_count += 1
            if assistant_turn == True:
                action_mask[pointer] = 1.0
                action_timestamps[pointer] = am_count 
            if token_ids[pointer] == 151645:
                assistant_turn = False
                
            pointer += 1
    else:
       raise TypeError("Tokenizer not supported. Must be implemented here.")

    return (
        action_mask, 
        action_timestamps,
        state_end_flags
    )
    