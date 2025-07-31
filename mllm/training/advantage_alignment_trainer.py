from mllm.training.reinforce_trainer import BaseTrainer
from dataclasses import dataclass
import torch
import numpy as np
from typing import Tuple
from markov_games.rollout_tree import RolloutTreeRootNode, ChatTurn, RolloutTreeBranchNode
from credit_methods import get_advantage_alignment_weights, advantages_to_aa_credits
from mllm.training.training_data_utils import get_main_chat_list_and_rewards
from mllm.training.tokenize_chats import process_training_chat
from mllm.training.training_data_utils import TrajectoryBatch, TrainingBatch
from mllm.training.training_data_utils import get_advantage_alignment_credits

RolloutId = int

class advantage_packet:
    agent_id: str
    rollout_ids: torch.IntTensor # (B,)
    advantages: torch.FloatTensor # (B, jT)

class AdAlignTrainer(BaseTrainer):
    """
    Extends the reinforce trainer to support Advantage Alignment.
    """

    def __init__(self, 
        ad_align_beta: float, 
        ad_align_gamma: float, 
        ad_align_include_k_equals_t: bool, 
        ad_align_use_sign: bool, 
        ad_align_clipping: float,
        ad_align_force_coop_first_step: bool,
        *args, **kwargs):
        """
        Initialize the advantage alignment trainer.
        Args:
            ad_align_beta: Beta parameter for the advantage alignment.
            ad_align_gamma: Gamma parameter for the advantage alignment.
            ad_align_include_k_equals_t: Whether to include k = t in the advantage alignment.
            ad_align_use_sign: Whether to use sign in the advantage alignment.
            ad_align_clipping: Clipping value for the advantage alignment.
            ad_align_force_coop_first_step: Whether to force coop on the first step of the advantage alignment.
        """
        super().__init__(*args, **kwargs)
        self.ad_align_beta = ad_align_beta
        self.ad_align_gamma = ad_align_gamma
        self.ad_align_include_k_equals_t = ad_align_include_k_equals_t
        self.ad_align_use_sign = ad_align_use_sign


    def set_pre_advantage_alignment_data(self, agent_ids: list[str], roots: list[RolloutTreeRootNode]):
        """
        TOWRITE
        Set the advantage alignment data for the trainer.
        """

        B = len(roots)

        # For main rollouts
        batch_rollout_ids = []
        batch_input_ids = []
        batch_action_mask = []
        batch_timesteps = []
        batch_state_ends_mask = []
        batch_rewards = []

        # For alternative actions rollouts
        alternative_batch_input_ids = []
        alternative_batch_action_mask = []
        alternative_batch_timesteps = []
        alternative_batch_state_ends_mask = []
        alternative_batch_rewards = []
        nb_alternative_actions = []

        for agent_id in agent_ids:

            for root in roots:

                batch_rollout_ids.append(root.id)
                main_chat, rewards = get_main_chat_list_and_rewards(agent_id=agent_id, root=root)
                (
                    input_ids,
                    action_mask,
                    timesteps,
                    state_ends_mask,
                ) = process_training_chat(
                    tokenizer=self.tokenizer,
                    chat_history=main_chat
                )

                # get the alternative trajectories of the rollout TODO: make sure to append existing history
                current_node = root.child
                chat = []
                rewards = []
                while current_node is not None:
                    assert isinstance(current_node, RolloutTreeBranchNode), "Current node must be a branch node"
                    branches = current_node.branches[agent_id]
                    nb_alternative_actions.append(len(branches)) # Number of alternative actions
                    for branch in branches:
                        chat, rewards = get_main_chat_list_and_rewards(agent_id=agent_id, root=branch)
                        (
                            input_ids,
                            action_mask,
                            timesteps,
                            state_ends_mask,
                        ) = process_training_chat(
                            tokenizer=self.tokenizer,
                            chat_history=chat
                        )
                        alternative_batch_input_ids.append(input_ids)
                        alternative_batch_action_mask.append(action_mask)
                        alternative_batch_timesteps.append(timesteps)
                        alternative_batch_state_ends_mask.append(state_ends_mask)
                        alternative_batch_rewards.append(rewards)

                    reward : float = current_node.step_log.simulation_step_log.rewards[agent_id]
                    rewards.append(reward)
                    chat_turns: list[ChatTurn] = current_node.step_log.action_logs[agent_id].chat_turns
                    chat.extend(chat_turns)
                    current_node = current_node.child 

                # Assert that number of alternative actions is constant
                assert len(set(nb_alternative_actions)) == 1, "Number of alternative actions must be constant"
                A = nb_alternative_actions[0]

                # Add main rollout to batch
                batch_input_ids.append(input_ids)
                batch_action_mask.append(action_mask)
                batch_timesteps.append(timesteps)
                batch_state_ends_mask.append(state_ends_mask)
                batch_rewards.append(rewards)

        trajectory_batch = TrajectoryBatch(
            rollout_ids = torch.Tensor(batch_rollout_ids), # (B,)
            batch_input_ids = torch.nested.nested_tensor(batch_input_ids, layout=torch.jagged), # (B, jS)
            batch_action_mask = torch.nested.nested_tensor(batch_action_mask, layout=torch.jagged), # (B, jS)
            batch_timesteps = torch.nested.nested_tensor(batch_timesteps, layout=torch.jagged), # (B, jS)
            batch_state_ends_mask = torch.nested.nested_tensor(batch_state_ends_mask, layout=torch.jagged), # (B, jS)
            batch_rewards = torch.nested.nested_tensor(batch_rewards, layout=torch.jagged) # (B, jT)
        )

        # Here, `A` is the number of alternative actions / trajectories taken at each time step. 
        alternative_trajectory_batch = TrajectoryBatch(
            rollout_ids = torch.zeros(B*A, dtype=torch.int32), # (B*A,) we don't have ids here
            batch_input_ids = torch.nested.nested_tensor(alternative_batch_input_ids, layout=torch.jagged), # (B*A, jS)
            batch_action_mask = torch.nested.nested_tensor(alternative_batch_action_mask, layout=torch.jagged), # (B*A, jS)
            batch_timesteps = torch.nested.nested_tensor(alternative_batch_timesteps, layout=torch.jagged), # (B*A, jS)
            batch_state_ends_mask = torch.nested.nested_tensor(alternative_batch_state_ends_mask, layout=torch.jagged), # (B*A, jS)
            batch_rewards = torch.nested.nested_tensor(alternative_batch_rewards, layout=torch.jagged) # (B*A, jT)
        )

        # Get Advantages & Train Critic
        self.batch_advantages: torch.FloatTensor = self.get_advantages_with_critic_gradient_accumulation(trajectory_batch) # (B, jT)
        self.batch_alternative_advantages: torch.FloatTensor = self.get_advantages_with_critic_gradient_accumulation(alternative_trajectory_batch) # (B*A, jT)
        self.batch_alternative_advantages = self.batch_alternative_advantages.reshape(B, A, -1) # (B, A, jT)

        # Update Critic
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad()

        # Send advantage packet to other players
        advantage_packet = advantage_packet(
            agent_id = self.agent_id,
            rollout_ids = self.trajectory_batch.rollout_ids,
            advantages = self.batch_advantages
        )
        return advantage_packet



    def set_advantage_alignment_data(self, advantage_packets: list[advantage_packet]):
        """
        Receive advantage packets from other players.
        These contain the advantages of the other players' rollouts estimated by them.
        """
        # TODO: do not assume a single agent is sending packet
        co_agent_packet = advantage_packets[0]
        co_agent_advantages = co_agent_packet.advantages
        co_agent_rollout_ids = co_agent_packet.rollout_ids
        assert self.batch_advantages.shape[0] == co_agent_advantages.shape[0], "Advantage shapes must match!"

        # Get co-agent advantages in the right order
        permutation = []
        for id in self.batch_advantages.rollout_ids: permutation.append(torch.where(id == co_agent_rollout_ids)[0].item())
        self.co_agent_advantages = torch.permute(co_agent_advantages, permutation)
        

        # Get padded tensors (advantage alignment is invariant to padding)
        jagged_lengths = self.batch_advantages.offsets().diff()
        padded_main_advantages = torch.nested.to_padded_tensor(self.batch_advantages, padding=0.0)
        padded_alternative_advantages = torch.nested.to_padded_tensor(self.batch_alternative_advantages, padding=0.0)
        padded_co_agent_advantages = torch.nested.to_padded_tensor(self.co_agent_advantages, padding=0.0)

        # Create training batch data
        credits = get_advantage_alignment_credits(
            a1 = padded_main_advantages,
            a1_star = padded_alternative_advantages,
            a2 = padded_co_agent_advantages,
        )
        advantage_alignment_credits = torch.nested.narrow(credits, dim=1, start=0, length=jagged_lengths, layout=torch.jagged)

        # Set training batch
        self.policy_gradient_data = TrainingBatch(
            rollout_ids = self.main_trajectories.rollout_ids,
            batch_input_ids = self.main_trajectories.batch_input_ids,
            batch_action_mask = self.main_trajectories.batch_action_mask,
            batch_credits = advantage_alignment_credits
        )







