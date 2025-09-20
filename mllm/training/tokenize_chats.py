import regex
import torch
from transformers import AutoTokenizer

from mllm.training.training_data_utils import TrainingChatTurn, TrajectoryBatch


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
        {%- if loop.index0 > ns.last_query_index %}
            {%- if reasoning_content %}
                {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n') }}
            {%- else %}
                {{- '<|im_start|>' + message.role + '\n' + content }}
            {%- endif %}
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

custom_gemma_template = """ {# Accepts: user / human / system / assistant / model #}
{%- set ns = namespace() -%}
{%- for message in messages %}
    {%- set role = (message.role or '') | lower -%}
    {%- set content = (message.content or '') -%}

    {# Map roles inside the template (no Python remapping needed) #}
    {%- if role in ['assistant', 'model'] -%}
        {%- set role_token = 'model' -%}
    {%- elif role in ['user', 'human'] -%}
        {%- set role_token = 'user' -%}
    {%- elif role == 'system' -%}
        {%- set role_token = 'user' -%}
        {%- set content = '[system]\n' ~ content -%}
    {%- else -%}
        {# Fallback: treat unknown roles as user #}
        {%- set role_token = 'user' -%}
    {%- endif -%}

    {{- '<start_of_turn>' ~ role_token ~ '\n' ~ content ~ '<end_of_turn>\n' -}}
{%- endfor %}
{%- if add_generation_prompt %}
{{- '<start_of_turn>model\n' -}}
{%- endif %}
"""  # original chat template for gemma at https://huggingface.co/google/gemma-3-4b-it/raw/main/tokenizer_config.json


def process_training_chat(
    tokenizer: AutoTokenizer,
    chat_history: list[TrainingChatTurn],
    entropy_mask_regex: str | None = None,
    exploration_prompts_to_remove: list[str] = [],
) -> tuple[torch.IntTensor, torch.BoolTensor, torch.IntTensor, torch.BoolTensor]:
    """Tokenize a single training chat and build aligned per-token masks.

    Given an ordered list of `TrainingChatTurn`, this function tokenizes each
    turn independently using the tokenizer's chat template, then concatenates
    all resulting token sequences. It also constructs three parallel 1D masks
    that align with the concatenated tokens:

    - input_ids: token ids for the entire chat, turn by turn
    - action_mask: True for tokens that belong to assistant turns (i.e., model
      actions), False for tokens from other roles
    - timesteps: per-token time step copied from the originating turn's
      `time_step`
    - state_ends_mask: True for the last token of any turn where
      `is_state_end` is True, otherwise False

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
        A tuple of four 1D tensors, all of equal length N (the total number of
        tokens across all turns), in the following order:
        - input_ids (LongTensor)
        - action_mask (BoolTensor)
        - timesteps (FloatTensor as implemented; see note above)
        - state_ends_mask (BoolTensor)
    """
    state_ends_mask = []
    input_ids = []
    action_mask = []
    timesteps = []
    entropy_mask = []
    for train_chat_turn in chat_history:
        is_state_end = train_chat_turn.is_state_end
        time_step = train_chat_turn.time_step
        is_action = train_chat_turn.role == "assistant" and (
            train_chat_turn.content != ""
        )

        # Remove exploration prompts from training data
        for exploration_prompt in exploration_prompts_to_remove:
            if exploration_prompt in train_chat_turn.content:
                train_chat_turn.content = train_chat_turn.content.replace(
                    exploration_prompt, ""
                )

        chat_turn = {
            "role": train_chat_turn.role,
            "content": train_chat_turn.content,
        }
        if entropy_mask_regex is not None:
            is_entropy_mask_true = (
                regex.search(entropy_mask_regex, train_chat_turn.content) is not None
            )
        else:
            is_entropy_mask_true = True

        # Tokenize (perhaps using a custom chat template)
        if "gemma" in tokenizer.name_or_path.lower():
            chat_turn_ids = tokenizer.apply_chat_template(
                [chat_turn], return_tensors="pt", chat_template=custom_gemma_template
            ).flatten()
        elif "qwen" in tokenizer.name_or_path.lower():
            chat_turn_ids = tokenizer.apply_chat_template(
                [chat_turn], return_tensors="pt", chat_template=custom_qwen_template
            ).flatten()
        else:
            chat_turn_ids = tokenizer.apply_chat_template(
                [chat_turn], return_tensors="pt"
            ).flatten()

        nb_chat_turns_ids = chat_turn_ids.numel()
        state_ends_mask.append(torch.zeros(nb_chat_turns_ids, dtype=torch.bool))
        if is_state_end:
            state_ends_mask[-1][-1] = True  # last token is state end
        input_ids.append(chat_turn_ids)
        action_mask.append(torch.ones(nb_chat_turns_ids, dtype=torch.bool))
        if not is_action:
            action_mask[-1] = action_mask[-1] * False
        entropy_mask.append(torch.ones(nb_chat_turns_ids, dtype=torch.bool))
        if not is_entropy_mask_true:
            entropy_mask[-1] = entropy_mask[-1] * False
        timesteps.append(torch.ones(nb_chat_turns_ids) * time_step)

    input_ids = torch.cat(input_ids)
    action_mask = torch.cat(action_mask)
    entropy_mask = torch.cat(entropy_mask)
    timesteps = torch.cat(timesteps)
    state_ends_mask = torch.cat(state_ends_mask)

    return (input_ids, action_mask, entropy_mask, timesteps, state_ends_mask)
