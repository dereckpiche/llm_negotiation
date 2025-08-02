from mllm.training.reinforce_trainer import BaseTrainer
from dataclasses import dataclass
import torch
import numpy as np
from typing import Tuple
from mllm.markov_games.rollout_tree import RolloutTreeRootNode, ChatTurn, RolloutTreeBranchNode
from mllm.training.training_data_utils import get_main_chat_list_and_rewards
from mllm.training.tokenize_chats import process_training_chat
from mllm.training.training_data_utils import TrajectoryBatch, TrainingBatch
from mllm.training.credit_methods import get_advantage_alignment_credits
import copy
import logging
import sys

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))

RolloutId = int

class AdvantagePacket:
    agent_id: str
    rollout_ids: torch.IntTensor # (B,)
    advantages: torch.FloatTensor # (B, jT)

def get_alternative_chat_histories(
    agent_id: str, 
    time_step: int, # The time_step of the branching we want the alt
    root : RolloutTreeRootNode) -> list[list[ChatTurn], list[torch.FloatTensor]]:
    """
    
    args:
        agent_id: The agent we want to get the chat history for.
        time_step: The time_step of the branching we want the alternative chat histories for.
        root: The root of the rollout tree.
    returns:
        alternative_chats: list[list[ChatTurn]]
        alternative_rewards: list[torch.FloatTensor]
    """
    current_node = root.child
    branches = current_node.branches
    main_chat = []
    for i in range(time_step):
        # TODO: We don't need to do this fully at each time step. It wastes so much compute. Just save the intermediate chat history.
        if current_node is not None:
            main_node = current_node.main_child
            branches = current_node.branches
            current_node = main_node.child
            chat_turns: list[ChatTurn] = main_node.step_log.action_logs[agent_id].chat_turns
            chat_turns = copy.copy(chat_turns)

            # This is crucial. We do not need to estimate the advantages before we branch out.  (This is already done when we process the main trajectory.)
            # We want the first estimated advantage that we return later to be the one for taking the alternative action.
            for chat_turn in chat_turns:
                chat_turn.is_state_end = False

            main_chat.extend(chat_turns)

    alternative_roots = branches[agent_id]
    alternative_chats = []
    alternative_rewards = []
    for alt_root in alternative_roots:
        chat, rewards = get_main_chat_list_and_rewards(agent_id=agent_id, root=alt_root)
        alternative_chats.append(chat)
        alternative_rewards.append(rewards)
    alternative_chats_with_main = [  + alt_chat for alt_chat in alternative_chats]
    return alternative_chats_with_main, alternative_rewards

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
        self.ad_align_clipping = ad_align_clipping
        self.ad_align_force_coop_first_step = ad_align_force_coop_first_step

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
        jT_list = []

        for agent_id in agent_ids:
            for root in roots:
                logger.info(f"Processing main trajectory of root {root.id} for agent {agent_id}")

                # Get main trajectory 
                batch_rollout_ids.append(root.id)
                main_chat, main_rewards = get_main_chat_list_and_rewards(agent_id=agent_id, root=root)
                (
                    input_ids,
                    action_mask,
                    timesteps,
                    state_ends_mask,
                ) = process_training_chat(
                    tokenizer=self.tokenizer,
                    chat_history=main_chat
                )
                batch_input_ids.append(input_ids)
                batch_action_mask.append(action_mask)
                batch_timesteps.append(timesteps)
                batch_state_ends_mask.append(state_ends_mask)
                batch_rewards.append(main_rewards)
                jT = main_rewards.numel() # TODO: better than this
                jT_list.append(jT)

                # Get all of the alternative trajectories in the tree
                logger.info(f"Processing alternative trajectory of root {root.id} for agent {agent_id}")
                for t in range(jT):
                    alternative_chats, alternative_rewards = get_alternative_chat_histories(agent_id=agent_id, time_step=t, root=root)
                    nb_alternative_actions.append(len(alternative_chats))
                    print(len(alternative_chats), len(alternative_rewards))
                    print(alternative_chats[0], alternative_rewards[0])
                    for chat, rewards in zip(alternative_chats, alternative_rewards):
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

        jT_list = torch.Tensor(jT_list)

        # Assert that number of alternative actions is constant
        assert len(set(nb_alternative_actions)) == 1, "Number of alternative actions must be constant"
        A = nb_alternative_actions[0]
        logger.info(f"Number of alternative actions: {A}")

        trajectory_batch = TrajectoryBatch(
            rollout_ids = torch.Tensor(batch_rollout_ids), # (B,)
            batch_input_ids = torch.nested.nested_tensor(batch_input_ids, layout=torch.jagged), # (B, jS)
            batch_action_mask = torch.nested.nested_tensor(batch_action_mask, layout=torch.jagged), # (B, jS)
            batch_timesteps = torch.nested.nested_tensor(batch_timesteps, layout=torch.jagged), # (B, jS)
            batch_state_ends_mask = torch.nested.nested_tensor(batch_state_ends_mask, layout=torch.jagged), # (B, jS)
            batch_rewards = torch.nested.nested_tensor(batch_rewards, layout=torch.jagged) # (B, jT)
        )

        # Here, `A` is the number of alternative actions / trajectories taken at each time step. 
        # For each of the `B` rollout, at each of its jT (`j` is for jagged, since each main rollout may be of a different length) steps, we take A alternate trajectories (from different actions).
        # Therefore, we have ∑jT * A trajectories to process. If each of the main trajectories have T steps, we will have `B*T*A` to process.

        alternative_trajectory_batch = TrajectoryBatch(
            rollout_ids = torch.zeros(B*A, dtype=torch.int32), # (B*A,) we don't have ids here
            batch_input_ids = torch.nested.nested_tensor(alternative_batch_input_ids, layout=torch.jagged), # (∑jT * A, jS')
            batch_action_mask = torch.nested.nested_tensor(alternative_batch_action_mask, layout=torch.jagged), # (∑jT * A, jS')
            batch_timesteps = torch.nested.nested_tensor(alternative_batch_timesteps, layout=torch.jagged), # (∑jT * A, jS')
            batch_state_ends_mask = torch.nested.nested_tensor(alternative_batch_state_ends_mask, layout=torch.jagged), # (∑jT * A, jS')
            batch_rewards = torch.nested.nested_tensor(alternative_batch_rewards, layout=torch.jagged) # (∑jT * A, jT')
        ) 

        # Get Advantages & Train Critic
        self.batch_advantages: torch.FloatTensor = self.get_advantages_with_critic_gradient_accumulation(trajectory_batch) # (B, jT)

        # Get alternative advantages
        self.batch_alternative_advantages: torch.FloatTensor = self.get_advantages_with_critic_gradient_accumulation(alternative_trajectory_batch) # (∑jT * A, jT')
        self.batch_alternative_advantages = self.batch_alternative_advantages[:, 0] # (∑jT * A,) # (we only want the advantages where we branched out)
        self.batch_alternative_advantages = torch.nested.narrow(self.batch_alternative_advantages, dim=0, start=0, length=jT_list, layout=torch.jagged) # (B*A, jT')
        self.batch_alternative_advantages = self.batch_alternative_advantages.reshape(B, A, -1) # (B, A, jT')

        # Update Critic
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad()

        # Send advantage packet to other players
        advantage_packet = AdvantagePacket(
            agent_id = self.agent_id,
            rollout_ids = self.trajectory_batch.rollout_ids,
            advantages = self.batch_advantages
        )
        return advantage_packet



    def set_advantage_alignment_data(self, advantage_packets: list[AdvantagePacket]):
        """
        Receive advantage packets from other players.
        These contain the advantages of the other players' rollouts estimated by them.
        """
        # TODO: do not assume a single agent is sending packet
        co_agent_packet = advantage_packets[0] # TODO: get packet of other agent 
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







