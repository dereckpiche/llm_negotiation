from transformers import AutoTokenizer
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


def get_assistant_actions_mask_and_score(
    tokenizer: AutoTokenizer,
    assistant_msg_scores: torch.Tensor, 
    token_ids:torch.Tensor):
    """
    Args:
        assistant_msg_scores:
            Score attributed to each assistant messages. Length is same
            as number of assistant messages in conversation. 
    """

    tokenizer_name = tokenizer.name_or_path
    token_ids = token_ids.squeeze()
    nb_assistant_messages = 0
    nb_tokens = token_ids.shape[0]
    scores = torch.zeros(token_ids.shape)
    action_mask = torch.zeros(token_ids.shape)
    if assistant_msg_scores.dim() == 0:
        assistant_msg_scores = assistant_msg_scores.unsqueeze(0)


    if tokenizer_name in ["Qwen/Qwen2.5-7B-Instruct", "Qwen2.5-0.5B-Instruct"]:
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
                am_count += 1
            if assistant_turn == True:
                scores[pointer] = assistant_msg_scores[am_count]
                action_mask[pointer] = 1.0
            if token_ids[pointer] == 151645:
                assistant_turn = False
            pointer += 1

        assert am_count == assistant_msg_scores.shape[0]-1, "Did not use all of the scores."
        
    else:
       raise TypeError("Tokenizer not supported. Must be implemented here.")

    return scores, action_mask
    