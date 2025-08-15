"""
https://github.com/huggingface/transformers/blob/v4.53.3/src/transformers/tokenization_utils_base.py#L1519
"""

import torch
from transformers import AutoTokenizer
from mllm.utils.tiny_utils import find_subsequence
from mllm.training.training_data_utils import TrainingChatTurn, TrajectoryBatch, ReasoningLimits


def get_chat_dicts(chat: list[TrainingChatTurn]) -> list[dict]:
    chat_dicts = [chat_turn.dict() for chat_turn in chat]
    return chat_dicts


# TODO: expand / test for different model classes
custom_qwen_template = """
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for message in messages %}
    {%- if message.content is string %}
        {%- set content = message.content %}
    {%- else %}
        {%- set content = '' %}
    {%- endif %}
    {%- if (message.role == "user") %}
        {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is string %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- else %}
            {%- if '</think>' in content %}
                {%- set reasoning_content = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') %}
                {%- set content = content.split('</think>')[-1].lstrip('\n') %}
            {%- endif %}
        {%- endif %}
        {%- if reasoning_content %}
            {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n') }}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\n' + content }}
        {%- endif %}
        {{- '<|im_end|>\n' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- endif %}
{%- endif %}
"""


def get_qwen_reasoning_limit_tuple(tokenizer: AutoTokenizer, chat_turn: TrainingChatTurn) ->  ReasoningLimits:
    """
    """
    encoded = tokenizer.apply_chat_template(
        [chat_turn], return_tensors=None, chat_template=custom_qwen_template, add_special_tokens=True
    )
    if chat_turn.role != "assistant" or chat_turn.reasoning_content is None: return None
    open_reasoning_ids = tokenizer.encode("\n<think>\n", add_special_tokens=False)
    close_reasoning_ids = tokenizer.encode("</think>\n\n", add_special_tokens=False)
    reasoning_start = find_subsequence(encoded, open_reasoning_ids)
    reasoning_end = find_subsequence(encoded, close_reasoning_ids) + len(close_reasoning_ids)
    assert reasoning_start != -1 and reasoning_end != -1 and reasoning_end > reasoning_start, f"Expected to find reasoning content in the assistant turn {tokenizer.decode(encoded)}"
    content_end = len(tokenizer.encode(chat_turn.content, add_special_tokens=False))
    return ReasoningLimits(reasoning_start, reasoning_end, content_end)


def process_training_chat(
    tokenizer: AutoTokenizer,
    chat_history: list[TrainingChatTurn],
) -> tuple[torch.IntTensor, torch.BoolTensor, torch.IntTensor, torch.BoolTensor, list[ReasoningLimits]]:
    """Tokenize a single training chat and build aligned per-token masks.

    Given an ordered list of `TrainingChatTurn`, this function tokenizes each
    turn independently using the tokenizer's chat template, then concatenates
    all resulting token sequences. It also constructs four parallel 1D masks
    that align with the concatenated tokens:

    - input_ids: token ids for the entire chat, turn by turn
    - action_mask: True for tokens that belong to assistant turns (i.e., model
      actions), False for tokens from other roles
    - timesteps: per-token time step copied from the originating turn's
      `time_step`
    - state_ends_mask: True for the last token of any turn where
      `is_state_end` is True, otherwise False
    - reasoning_limit_tuples: list of tuples (start, end) of the reasoning blocks

    Important details:
    - Each turn is passed as a single-message list to
      `tokenizer.apply_chat_template` and flattened; the per-turn outputs are
      then concatenated in the original order.
    - Turn boundaries are not explicitly encoded beyond what the chat template
      inserts; masks provide alignment for learning signals and state endings.
    - No truncation or padding is performed here; downstream code should handle
      batching/padding as needed.
    - Note on dtypes: `input_ids` will be a LongTensor (int64). `action_mask`
      and `state_ends_mask` are BoolTensors. `timesteps` is currently created
      as a float tensor; adjust the implementation if integer dtype is
      required downstream.

    Args:
        tokenizer: A Hugging Face tokenizer supporting `apply_chat_template`.
        chat_history: Ordered list of `TrainingChatTurn` forming one dialogue.

    Returns:
        A tuple of five 1D tensors, all of equal length N (the total number of
        tokens across all turns), in the following order:
        - input_ids (LongTensor)
        - action_mask (BoolTensor)
        - timesteps (FloatTensor as implemented; see note above)
        - state_ends_mask (BoolTensor)
        - reasoning_limit_tuples (list[tuple[int, int]])
    """
    state_ends_mask = []
    input_ids = []
    action_mask = []
    timesteps = []
    reasoning_limit_tuples = []
    token_counter = 0
    for train_chat_turn in chat_history:
        is_state_end = train_chat_turn.is_state_end
        time_step = train_chat_turn.time_step
        is_action = train_chat_turn.role == "assistant"
        chat_turn = {
            "role": train_chat_turn.role,
            "content": train_chat_turn.content,
            "reasoning_content": train_chat_turn.reasoning_content,
        }
        chat_turn_ids = tokenizer.apply_chat_template(
            [chat_turn], return_tensors="pt", chat_template=custom_qwen_template
        ).flatten()
        nb_chat_turns_ids = chat_turn_ids.numel()
        state_ends_mask.append(torch.zeros(nb_chat_turns_ids, dtype=torch.bool))
        if is_state_end:
            state_ends_mask[-1][-1] = True  # last token is state end
        reasoning_limit_tuple = get_qwen_reasoning_limit_tuple(tokenizer, train_chat_turn)
        if reasoning_limit_tuple is not None:
            reasoning_limit_tuple.reasoning_start += token_counter
            reasoning_limit_tuple.reasoning_end += token_counter
            reasoning_limit_tuple.content_end += token_counter
            reasoning_limit_tuples.append(reasoning_limit_tuple)
        input_ids.append(chat_turn_ids)
        action_mask.append(torch.ones(nb_chat_turns_ids, dtype=torch.bool))
        if not is_action:
            action_mask[-1] = action_mask[-1] * False
        timesteps.append(torch.ones(nb_chat_turns_ids) * time_step)
        token_counter += nb_chat_turns_ids


    input_ids = torch.cat(input_ids)
    action_mask = torch.cat(action_mask)
    timesteps = torch.cat(timesteps)
    state_ends_mask = torch.cat(state_ends_mask)
    return (input_ids, action_mask, timesteps, state_ends_mask, reasoning_limit_tuples)




