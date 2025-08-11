import torch
from transformers import AutoTokenizer

from mllm.training.training_data_utils import TrainingChatTurn, TrajectoryBatch


def get_chat_dicts(chat: list[TrainingChatTurn]) -> list[dict]:
    chat_dicts = [chat_turn.dict() for chat_turn in chat]
    return chat_dicts


# TODO: expand / test for different model classes
custom_qwen_template = """{% for m in messages -%}
<|im_start|>{{ m['role'] }}
{{ m['content'] }}<|im_end|>
{% endfor -%}
{% if add_generation_prompt -%}
<|im_start|>assistant
{%- endif %}"""


def process_training_chat(
    tokenizer: AutoTokenizer,
    chat_history: list[TrainingChatTurn],
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
    include_system_prompt = True
    for train_chat_turn in chat_history:
        is_state_end = train_chat_turn.is_state_end
        time_step = train_chat_turn.time_step
        is_action = train_chat_turn.role == "assistant"
        chat_turn = {
            "role": train_chat_turn.role,
            "content": train_chat_turn.content,
        }
        chat_turn_ids = tokenizer.apply_chat_template(
            [chat_turn], return_tensors="pt", chat_template=custom_qwen_template
        ).flatten()
        nb_chat_turns_ids = chat_turn_ids.numel()
        state_ends_mask.append(torch.zeros(nb_chat_turns_ids, dtype=torch.bool))
        if is_state_end:
            state_ends_mask[-1][-1] = True  # last token is state end
        input_ids.append(chat_turn_ids)
        action_mask.append(torch.ones(nb_chat_turns_ids, dtype=torch.bool))
        if not is_action:
            action_mask[-1] = action_mask[-1] * False
        timesteps.append(torch.ones(nb_chat_turns_ids) * time_step)
        include_system_prompt = False

    input_ids = torch.cat(input_ids)
    action_mask = torch.cat(action_mask)
    timesteps = torch.cat(timesteps)
    state_ends_mask = torch.cat(state_ends_mask)

    return (input_ids, action_mask, timesteps, state_ends_mask)
