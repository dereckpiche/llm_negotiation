"""
TODO: this code is terrible and needs to be improved.
TODO: Use the return_assistant_tokens_mask feature from  https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizer.apply_chat_template
TODO: why are we keepign trakc of current_time_step?
Hack :Qwen2.5 has system prompt even when its explicitly set to False
Hack : <think>\n\n</think>\n\n is present in Qwen3-4B-Instruct-2507 without thinking mode
"""


import torch
from transformers import AutoTokenizer

from mllm.markov_games.rollout_tree import ChatTurn
from mllm.training.training_data_utils import TrajectoryBatch


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
        return_tensors="pt",
    ).tolist()[0]
    sentencepieces = tokenizer.convert_ids_to_tokens(token_ids)
    sps = []
    for tid, sp in zip(token_ids, sentencepieces):
        sps.append((tid, sp))
    return sps


def get_qwen_assistant_user_mask(
    tokenizer: AutoTokenizer,
    token_ids: torch.Tensor,
    has_system_prompt: bool = False,
):
    """
    Returns:
        assistant_user_mask:
            assistant_user_mask[i] = -k means that the i'th token id belongs to the
            (k)'th user message
            assistant_user_mask[i] = k means that the i'th token id belongs to the
            (k)'th assistant message
            assistant_user_mask[i] = 0 means that the i'th token id belongs to the
            system prompt.
    For this tokenizer, eos_token_id is 151645 and get_sentencepieced_example(qwen_tokenizer) returns

        [(151644, '<|im_start|>'), (8948, 'system'), (198, 'Ċ'), (2610, 'You'), (525, 'Ġare'), (1207, 'ĠQ'), (16948, 'wen'), (11, ','), (3465, 'Ġcreated'), (553, 'Ġby'), (54364, 'ĠAlibaba'), (14817, 'ĠCloud'), (13, '.'), (1446, 'ĠYou'), (525, 'Ġare'), (264, 'Ġa'), (10950, 'Ġhelpful'), (17847, 'Ġassistant'), (13, '.'), (151645, '<|im_end|>'), (198, 'Ċ'), (151644, '<|im_start|>'), (872, 'user'), (198, 'Ċ'), (1112, '...'), (151645, '<|im_end|>'), (198, 'Ċ'), (151644, '<|im_start|>'), (77091, 'assistant'), (198, 'Ċ'), (1112, '...'), (151645, '<|im_end|>'), (198, 'Ċ'), (151644, '<|im_start|>'), (872, 'user'), (198, 'Ċ'), (1112, '...'), (151645, '<|im_end|>'), (198, 'Ċ'), (151644, '<|im_start|>'), (77091, 'assistant'), (198, 'Ċ'), (1112, '...'), (151645, '<|im_end|>'), (198, 'Ċ')]

    """
    eos_token_id = 151645
    bos_token_id = 151644
    assistant_token_id = 77091
    user_token_id = 872

    nb_tokens = token_ids.shape[0]
    assistant_count = 0
    user_count = 0
    assistant_turn = False
    pointer = 0
    assistant_user_mask = torch.full(token_ids.shape, 0)

    system_prompt_end = -1
    if has_system_prompt:
        # skip and put inf for system prompt
        system_prompt_end = torch.where(token_ids == eos_token_id)[0][0].item()
        assistant_user_mask[0:system_prompt_end] = 0

    pointer = system_prompt_end + 1
    while pointer < nb_tokens:
        # new user turn
        if (
            token_ids[pointer] == bos_token_id
            and token_ids[pointer + 1] == user_token_id
        ):
            assistant_turn = False
            user_count += 1
        # new assistant turn
        if (
            token_ids[pointer] == bos_token_id
            and token_ids[pointer + 1] == assistant_token_id
        ):
            assistant_user_mask[pointer : pointer + 3] = -user_count
            pointer += 3
            assistant_turn = True
            assistant_count += 1
        if assistant_turn == True:
            assistant_user_mask[pointer] = assistant_count
        else:
            assistant_user_mask[pointer] = -user_count
        pointer += 1
    return assistant_user_mask


def get_chat_dicts(chat: list[ChatTurn]) -> list[dict]:
    chat_dicts = [chat_turn.dict() for chat_turn in chat]
    return chat_dicts


def process_training_chat(
    tokenizer: AutoTokenizer,
    chat_history: list[ChatTurn],
    end_at_last_state_flag: bool = False,
):
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
    # TODO: clean up!

    chat_history = get_chat_dicts(chat_history)
    # End chat on the last state introduction
    if end_at_last_state_flag:
        count = 0
        for message in chat_history:
            if message["is_state_end"]:
                last_state_flag_index = count
            count += 1
        chat_history = chat_history[: last_state_flag_index + 1]

    # Get token ids
    formatted_conversation = tokenizer.apply_chat_template(
        chat_history,
        add_generation_prompt=False,
        tokenize=False,
    )
    tokenizer_name = tokenizer.name_or_path
    if "Qwen/Qwen3-4B-Instruct-2507" in tokenizer_name:
        formatted_conversation = formatted_conversation.replace(
            "<think>\n\n</think>\n\n", ""
        )
    token_ids = torch.tensor(tokenizer.encode(formatted_conversation), dtype=torch.long)

    has_system_prompt = False
    if "Qwen2.5" in tokenizer_name:
        has_system_prompt = True
    # Get assistant_user_mask
    if tokenizer_name in [
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen3-4B-Instruct-2507",
        "Qwen/Qwen3-0.6B",
    ]:
        assistant_user_mask = get_qwen_assistant_user_mask(
            tokenizer=tokenizer,
            token_ids=token_ids,
            has_system_prompt=has_system_prompt,
        )
    else:
        raise TypeError("Tokenizer not supported. Must be implemented here.")

    # Create masks and flags
    action_mask = torch.zeros(size=token_ids.shape)
    credit_mask = torch.full(size=token_ids.shape, fill_value=-1.0)
    state_end_flags = torch.full(size=token_ids.shape, fill_value=False)

    assistant_count = 1
    user_count = -1
    current_time_step = -1

    for message in chat_history:
        if message["role"] == "user":
            if message["is_state_end"]:
                # Get index of first token with value = user_count
                state_end_flag = torch.argmax(
                    (assistant_user_mask == -user_count).int()
                ).item()
                state_end_flags[state_end_flag] = True
            user_count -= 1
        if message["role"] == "assistant":
            time_step = message["time_step"]
            if current_time_step < time_step:
                current_time_step = time_step
            credit_mask[assistant_user_mask == assistant_count] = current_time_step
            assistant_count += 1

    action_mask[credit_mask > -1] = 1.0

    input_ids = token_ids
    timesteps = credit_mask
    state_ends_idx = (
        state_end_flags  # torch.where(torch.BoolTensor(state_end_flags) == True)[0]
    )

    return (input_ids, action_mask, timesteps, state_ends_idx)
