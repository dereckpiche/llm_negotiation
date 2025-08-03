import torch
from dataclasses import dataclass
import torch.nested as tn
from typing import Literal, Tuple
from mllm.markov_games.rollout_tree import RolloutTreeNode, RolloutTreeBranchNode, RolloutTreeRootNode, ChatTurn

class TrainingChatTurn:
    # TODO: simplify by making this a child of ChatTurn
    """
    This class contains the chat turns for a single agent. 
    It is like ChatTurn, but with the time step added.
    """ 
    def __init__(self, time_step: int, role: str, agent_id: str, content: str, is_state_end: bool):
        self.time_step = time_step
        self.role = role
        self.agent_id = agent_id
        self.content = content
        self.is_state_end = is_state_end
    def dict(self):
        return {
            "time_step": self.time_step,
            "role": self.role,
            "agent_id": self.agent_id,
            "content": self.content,
            "is_state_end": self.is_state_end
        }


def get_main_chat_list_and_rewards(
    agent_id: str, root : RolloutTreeRootNode | RolloutTreeNode) -> Tuple[list[TrainingChatTurn], torch.FloatTensor]:
    """
    This method traverses a rollout tree and returns a the list of ChatTurn
    for an agent. If it encounters a branch node, it follows the main path.
    """
    # TODO; extend for all trees, not just linear
    if isinstance(root, RolloutTreeRootNode):
        current_node = root.child
    else:
        current_node = root

    chat = []
    rewards = []
    while current_node is not None:
        if isinstance(current_node, RolloutTreeBranchNode):
            current_node = current_node.main_child
        reward : float = current_node.step_log.simulation_step_log.rewards[agent_id]
        rewards.append(reward)
        chat_turns: list[TrainingChatTurn] = current_node.step_log.action_logs[agent_id].chat_turns
        chat_turns = [TrainingChatTurn(time_step=current_node.time_step, **turn.model_dump()) for turn in chat_turns]
        chat.extend(chat_turns)
        current_node = current_node.child
    return chat, torch.FloatTensor(rewards)


def get_tokenwise_credits(
    # B := batch size, S := number of tokens / seq. length, T := number of states. `j` stands for jagged (see pytorch nested tensors.)
    batch_timesteps:      torch.IntTensor | torch.Tensor, # (B, jS),
    batch_credits:        torch.FloatTensor | torch.Tensor # (B, jT)
    ) -> torch.FloatTensor | torch.Tensor: # (B, jS)
    """
    TOWRITE
    """
    #TODO vectorize this code
    batch_token_credits = []
    for credits, timesteps in zip(batch_credits, batch_timesteps):
        token_credits = torch.zeros_like(timesteps, dtype=credits.dtype, device=timesteps.device,)
        for idx, credit in enumerate(credits):
            token_credits[timesteps == idx] = credit
        batch_token_credits.append(token_credits)
    batch_token_credits = tn.nested_tensor(batch_token_credits, layout=torch.jagged)
    return batch_token_credits


@dataclass
class TrajectoryBatch:
    """
    Tensorized batch of trajectories.
    """
    # B := batch size, S := number of tokens / seq. length, T := number of states. `j` stands for jagged (see pytorch nested tensors.)
    rollout_ids: torch.IntTensor # (B,)
    batch_input_ids:      torch.LongTensor | torch.Tensor # (B, jS)
    batch_action_mask:    torch.BoolTensor | torch.Tensor # (B, jS)
    batch_timesteps:      torch.IntTensor | torch.Tensor # (B, jS)
    batch_state_ends_mask: torch.IntTensor | torch.Tensor # (B, jS)
    batch_rewards:        torch.FloatTensor | torch.Tensor # (B, jT)

    def __post_init__(self):
        """
        This method is executed after initialization automatically.
        It ensures that the tensors created match the requirements.
        """
        B = self.rollout_ids.shape[0]
        if self.batch_input_ids.dim() == 1:
            self.batch_input_ids = self.batch_input_ids.unsqueeze(0)
            self.batch_action_mask = self.batch_action_mask.unsqueeze(0)
            self.batch_timesteps = self.batch_timesteps.unsqueeze(0)
            self.batch_state_ends_mask = self.batch_state_ends_mask.unsqueeze(0)
            self.batch_rewards = self.batch_rewards.unsqueeze(0)
        for b in range(B):
            nb_rewards = self.batch_rewards[b].shape[0]
            nb_timesteps = torch.max(self.batch_timesteps[b]).item() + 1
            print(nb_rewards, nb_timesteps)
            assert nb_rewards == nb_timesteps, "Number of rewards and timesteps mismatch."
            assert self.batch_input_ids[b].shape[0] == self.batch_action_mask[b].shape[0] == self.batch_timesteps[b].shape[0], "Tensors must have the same shape along the jagged dimension."
            assert self.batch_state_ends_mask[b].sum() == self.batch_rewards[b].shape[0], "Number of rewards must match number of state ends."


    """
    Entries:
        Here, we ignore the batch dimension.
        input_ids:
            All of the tokens of both the user and the assistant, flattened.
        action_mask:
            Set to true on the tokens of the assistant (tokens generated by the model).
        timesteps:
            Therefore, max(timesteps) = Ns - 1.
        state_ends_idx:
            Indices of the tokens at which state descriptions end.
        rewards:
            rewards[t] := R_t(s_t, a_t)
    Example:
        position:       "0  1  2  3  4  5  6  7  8  9  10 11 12 13 14"
        input_ids:      "U  U  U  a  a  a  U  a  U  a  a  a  U  U  U" (U := User, a := Assistant)
        action_mask:    "x  x  x  ✓  ✓  ✓  x  ✓  x  ✓  ✓  ✓  x  x  x"
        timestep:       "0  0  0  0  0  0  1  1  1  1  1  1  2  2  2"
        state_ends_dx:  [2, 6, 14]
        rewards:        [r0, r1, r2] 
    """
    def __getitem__(self, key) -> "TrajectoryBatch":
        if isinstance(key, slice):
            ret = TrajectoryBatch(
                rollout_ids = self.rollout_ids.__getitem__(key),
                batch_input_ids = tn.nested_tensor(self.batch_input_ids.unbind().__getitem__(key), layout=torch.jagged),
                batch_action_mask = tn.nested_tensor(self.batch_action_mask.unbind().__getitem__(key), layout=torch.jagged),
                batch_timesteps = tn.nested_tensor(self.batch_timesteps.unbind().__getitem__(key), layout=torch.jagged),
                batch_state_ends_mask = tn.nested_tensor(self.batch_state_ends_mask.unbind().__getitem__(key), layout=torch.jagged),
                batch_rewards = tn.nested_tensor(self.batch_rewards.unbind().__getitem__(key), layout=torch.jagged)
            )
            return ret

    def __len__(self):
        return self.batch_input_ids.shape[0]

    def to(self, device):
        self.rollout_ids = self.rollout_ids.to(device)
        self.batch_input_ids = self.batch_input_ids.to(device)
        self.batch_action_mask = self.batch_action_mask.to(device)
        self.batch_timesteps = self.batch_timesteps.to(device)
        self.batch_state_ends_mask = self.batch_state_ends_mask.to(device)
        self.batch_rewards = self.batch_rewards.to(device)

    def get_padded_tensors_for_critic(self):
        """
        TOWRITE
        """
        padded_batch_input_ids = tn.to_padded_tensor(self.batch_input_ids, padding=0.0)
        padded_batch_state_ends_mask = tn.to_padded_tensor(self.batch_state_ends_mask, padding=False)
        jagged_lengths = self.batch_input_ids.offsets().clone()
        jagged_lengths[1:] = jagged_lengths[1:] - jagged_lengths[:-1] # TODO: verify
        jagged_lengths = jagged_lengths[1:]
        return padded_batch_input_ids, padded_batch_state_ends_mask, jagged_lengths

timestep = int


@dataclass
class PaddedTensorTrainingBatch:
    batch_input_ids:      torch.LongTensor | torch.Tensor
    batch_action_mask:    torch.BoolTensor | torch.Tensor
    batch_credits:        torch.FloatTensor | torch.Tensor
    def __len__(self):
        return self.batch_input_ids.shape[0]
    def to(self, device):
        self.batch_input_ids = self.batch_input_ids.to(device)
        self.batch_action_mask = self.batch_action_mask.to(device)
        self.batch_credits = self.batch_credits.to(device)

@dataclass
class TrainingBatch:
    rollout_ids: torch.IntTensor | torch.Tensor # (B,)
    batch_input_ids:      torch.LongTensor | torch.Tensor # (B, jS)
    batch_action_mask:    torch.FloatTensor | torch.Tensor # (B, jS)
    batch_credits: torch.FloatTensor | torch.Tensor # (B, jS)

    def __post_init__(self):
        # Put everything in the right device
        self.rollout_ids = self.rollout_ids.to("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_input_ids = self.batch_input_ids.to("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_action_mask = self.batch_action_mask.to("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_credits = self.batch_credits.to("cuda" if torch.cuda.is_available() else "cpu")
        # Ensure batch dimension is present
        assert self.batch_input_ids.dim() == self.batch_action_mask.dim() == self.batch_credits.dim() == 2, "Tensors must be of shape (B,jS)"
        assert self.batch_input_ids.shape[0] == self.batch_action_mask.shape[0] == self.batch_credits.shape[0], "Tensors must have the same batch size."
        input_diff = self.batch_input_ids.offsets().diff()
        action_diff = self.batch_action_mask.offsets().diff()
        credit_diff = self.batch_credits.offsets().diff()
        assert torch.all(input_diff == action_diff).item() and torch.all(action_diff == credit_diff).item(), \
            "Tensors must have the same shapes along the jagged dimension."

    def __getitem__(self, key) -> "TrainingBatch":
        if isinstance(key, slice):
            ret = TrainingBatch(
                rollout_ids = self.rollout_ids.__getitem__(key),
                batch_input_ids = tn.nested_tensor(self.batch_input_ids.unbind().__getitem__(key), layout=torch.jagged),
                batch_action_mask = tn.nested_tensor(self.batch_action_mask.unbind().__getitem__(key), layout=torch.jagged),
                batch_credits = tn.nested_tensor(self.batch_credits.unbind().__getitem__(key), layout=torch.jagged)
            )
            return ret

    def __len__(self):
        return self.batch_input_ids.shape[0]

    def to(self, device):
        self.rollout_ids = self.rollout_ids.to(device)
        self.batch_input_ids = self.batch_input_ids.to(device)
        self.batch_action_mask = self.batch_action_mask.to(device)
        self.batch_credits = self.batch_credits.to(device)

    def get_padded_tensors(self, padding: float = 0.0):
        """
        TOWRITE
        Always pad to the right.
        """
        padded_batch_input_ids = tn.to_padded_tensor(self.batch_input_ids, padding=padding)
        padded_batch_action_mask = tn.to_padded_tensor(self.batch_action_mask, padding=padding)
        padded_batch_credits = tn.to_padded_tensor(self.batch_credits, padding=padding)



        return PaddedTensorTrainingBatch(padded_batch_input_ids, padded_batch_action_mask, padded_batch_credits)

    def append(self, other: "TrainingBatch"):
        self.rollout_ids = torch.cat([self.rollout_ids, other.rollout_ids])
        self.batch_input_ids = tn.cat([self.batch_input_ids, other.batch_input_ids], layout=torch.jagged)
        self.batch_action_mask = tn.cat([self.batch_action_mask, other.batch_action_mask], layout=torch.jagged)
        self.batch_credits = tn.cat([self.batch_credits, other.batch_credits], layout=torch.jagged)

timestep = int
