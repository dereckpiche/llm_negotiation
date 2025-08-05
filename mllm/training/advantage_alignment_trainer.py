from mllm.training.reinforce_trainer import BaseTrainer
from dataclasses import dataclass
import torch
import numpy as np
from typing import Tuple
from mllm.markov_games.rollout_tree import RolloutTreeRootNode, ChatTurn, RolloutTreeBranchNode
from mllm.training.training_data_utils import get_main_chat_list_and_rewards, TrainingChatTurn
from mllm.training.tokenize_chats import process_training_chat
from mllm.training.training_data_utils import TrajectoryBatch, get_tokenwise_credits, TrainingBatch
from mllm.training.credit_methods import get_advantage_alignment_credits
import copy
import logging
import sys

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))

RolloutId = int
AgentId = str

@dataclass
class AdAlignTrainingData:
    agent_id: str
    main_data: TrajectoryBatch
    main_advantages: torch.FloatTensor | None = None # (B, jT) where B is the number of rollouts and jT is the number of time steps in the main trajectory
    alternative_advantages: torch.FloatTensor | None = None # (B, jT, A) where A is the number of alternative actions
    advantage_alignment_credits: torch.FloatTensor | None = None # (B, jS) where B is the number of rollouts and jS is the number of tokens

@dataclass
class AdvantagePacket:
    agent_id: str
    rollout_ids: torch.IntTensor # (B,)
    main_advantages: torch.FloatTensor # (B, jT)

def get_alternative_chat_histories(
    agent_id: str, 
    root : RolloutTreeRootNode) -> list[list[TrainingChatTurn], list[torch.FloatTensor]]:
    """
    args:
        agent_id: The agent we want to get the chat history for.
        root: The root of the rollout tree.
    returns:
        alternative_chats: list[list[TrainingChatTurn]] (jT*A, jS')
        alternative_rewards: list[torch.FloatTensor] (jT*A, jT')
    """
    current_node = root.child
    branches = current_node.branches
    pre_branch_chat = []
    pre_branch_rewards = []
    alternative_rewards = []
    alternative_chats = []
    while current_node is not None:
        assert isinstance(current_node, RolloutTreeBranchNode), "Current node should be a branch node."
        main_node = current_node.main_child
        branches = current_node.branches
        current_node = main_node.child
        
        # Get the `A` alternative trajectories
        alternative_nodes = branches[agent_id]
        for alt_node in alternative_nodes:
            post_branch_chat, post_branch_rewards = get_main_chat_list_and_rewards(agent_id=agent_id, root=alt_node)
            print(100*"-")
            print("\n\nPre branch chat\n\n")
            for turn in pre_branch_chat:
                print(turn.dict())
            print("\n\nPost branch chat\n\n")
            for turn in post_branch_chat:
                print(turn.dict())
            print(100*"-")

            branch_chat = pre_branch_chat + post_branch_chat
            alternative_chats.append(branch_chat)
            alternative_rewards.append(torch.cat([torch.tensor(pre_branch_rewards),post_branch_rewards]))

        chat_turns: list[ChatTurn] = main_node.step_log.action_logs[agent_id].chat_turns
        chat_turns: list[TrainingChatTurn] = [TrainingChatTurn(time_step=main_node.time_step, **turn.model_dump()) for turn in chat_turns]

        pre_branch_chat.extend(chat_turns)
        pre_branch_rewards.append(main_node.step_log.simulation_step_log.rewards[agent_id])

    return alternative_chats, alternative_rewards

class AdAlignTrainer(BaseTrainer):
    """
    Extends the reinforce trainer to support Advantage Alignment.
    """

    def __init__(self, 
        ad_align_beta: float, 
        ad_align_gamma: float, 
        ad_align_exclude_k_equals_t: bool, 
        ad_align_use_sign: bool, 
        ad_align_clipping: float,
        ad_align_force_coop_first_step: bool,
        *args, **kwargs):
        """
        Initialize the advantage alignment trainer.
        Args:
            ad_align_beta: Beta parameter for the advantage alignment.
            ad_align_gamma: Gamma parameter for the advantage alignment.
            ad_align_exclude_k_equals_t: Whether to include k = t in the advantage alignment.
            ad_align_use_sign: Whether to use sign in the advantage alignment.
            ad_align_clipping: Clipping value for the advantage alignment.
            ad_align_force_coop_first_step: Whether to force coop on the first step of the advantage alignment.
        """
        super().__init__(*args, **kwargs)
        self.ad_align_beta = ad_align_beta
        self.ad_align_gamma = ad_align_gamma
        self.ad_align_exclude_k_equals_t = ad_align_exclude_k_equals_t
        self.ad_align_use_sign = ad_align_use_sign
        self.ad_align_clipping = ad_align_clipping
        self.ad_align_force_coop_first_step = ad_align_force_coop_first_step
        self.training_data: dict[AgentId, AdAlignTrainingData]  = {} 

    def set_pre_advantage_alignment_data(self, agent_id: str, roots: list[RolloutTreeRootNode]):
        """
        TOWRITE
        Set the advantage alignment data for the trainer.
        """

        B = len(roots) # Number of rollouts

        # For main rollouts
        batch_rollout_ids = []
        batch_input_ids = []
        batch_action_mask = []
        batch_timesteps = []
        batch_state_ends_mask = []
        batch_rewards = []

        # For alternative actions rollouts
        batch_branching_time_steps = []
        alternative_batch_input_ids = []
        alternative_batch_action_mask = []
        alternative_batch_timesteps = []
        alternative_batch_state_ends_mask = []
        alternative_batch_rewards = []
        jT_list = []

        A = len(roots[0].child.branches[agent_id]) # Number of alternative actions

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
            logger.info(f"Main trajectory length (jT): {jT}")
            jT_list.append(jT)

            # We get the branching time steps for each of the `jT` time steps in the main trajectory.
            batch_branching_time_steps.extend(range(jT))

            # Get all of the (jT*A) alternative trajectories in the tree 
            # (jT is the number of time steps in the main trajectory, A is the number of alternative actions)
            logger.info(f"Processing alternative trajectory of root {root.id} for agent {agent_id}")
            alternative_chats, alternative_rewards = get_alternative_chat_histories(agent_id=agent_id, root=root)
            assert len(alternative_chats) == A * jT, "Incorrect number of alternative trajectories."

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
        # assert len(set(nb_alternative_actions)) == 1, "Number of alternative actions must be constant"
        # A = nb_alternative_actions[0]
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
        # For each of the `B` rollout perspectives, at each of its jT (`j` is for jagged, since each main rollout may be of a different length) steps, we take A alternate trajectories (from different actions).
        # Therefore, we have ∑jT * A trajectories to process. If each of the main trajectories have T steps, we will have `B*T*A` to process.
        sum_jT = int(torch.sum(jT_list).item())
        jT_list = jT_list.int().tolist() # (jT,) # (we only want the advantages where we branched out)
        alternative_trajectory_batch = TrajectoryBatch(
            rollout_ids =  torch.zeros(A*sum_jT, dtype=torch.int32), # (B*A,) we don't have ids here
            batch_input_ids = torch.nested.nested_tensor(alternative_batch_input_ids, layout=torch.jagged), # (∑jT * A, jS')
            batch_action_mask = torch.nested.nested_tensor(alternative_batch_action_mask, layout=torch.jagged), # (∑jT * A, jS')
            batch_timesteps = torch.nested.nested_tensor(alternative_batch_timesteps, layout=torch.jagged), # (∑jT * A, jS')
            batch_state_ends_mask = torch.nested.nested_tensor(alternative_batch_state_ends_mask, layout=torch.jagged), # (∑jT * A, jS')
            batch_rewards = torch.nested.nested_tensor(alternative_batch_rewards, layout=torch.jagged) # (∑jT * A, jT')
        ) 
        batch_branching_time_steps = torch.Tensor(batch_branching_time_steps).to(dtype=torch.int64, device=self.device) # (∑jT * A, 1)

        # Get Advantages & Train Critic
        self.batch_advantages: torch.FloatTensor = self.get_advantages_with_critic_gradient_accumulation(trajectory_batch) # (B, jT)
        logger.info(f"Batch advantages shape (B, jT): {self.batch_advantages.shape[0]}, jT=  {self.batch_advantages.offsets().diff()}")
        logger.info(f"JT list: {jT_list}")


        # Get alternative advantages
        # BAAs stands for batch alternative advantages
        # (torch nested tensors have very little api support, so we have to do some odd manual work here)
        BAAs: torch.FloatTensor = self.get_advantages_with_critic_gradient_accumulation(alternative_trajectory_batch) # (∑jT * A, jT')
        BAAs = torch.nested.to_padded_tensor(BAAs, padding=0.0) # (∑jT * A, P) # necessary for slice operations
        BAAs = torch.gather(BAAs, dim=1, index=batch_branching_time_steps.unsqueeze(1)) # (∑jT * A,) # (we only want the advantages where we branched out)
        BAAs = torch.nested.nested_tensor( [chunk for block in BAAs.view(A, sum_jT) for chunk in block.split(jT_list)], layout=torch.jagged ) # (B*A, jT)
        BAAs = torch.nested.to_padded_tensor(BAAs, padding=0.0) # (B*A, P) # necessary for reshape operation
        BAAs = BAAs.reshape(B, -1, A) # (B, P, A)
        BAAs = torch.nested.nested_tensor([block[:max] for max, block in zip(jT_list, BAAs)], layout=torch.jagged) # (B, jT, A)

        self.training_data[agent_id] = AdAlignTrainingData(
            agent_id = agent_id,
            main_data = trajectory_batch,
            main_advantages = self.batch_advantages,
            alternative_advantages = BAAs,
        )


    def share_advantage_alignment_data(self) -> AdvantagePacket:
        """
        Share the advantage alignment data with other agents.
        Returns:
            AdvantagePacket: The advantage packet containing the agent's advantages.
        """
        advantage_packets = []
        for _, agent_data in self.training_data.items():
            advantage_packets.append(AdvantagePacket(
                agent_id = agent_data.agent_id,
                rollout_ids = agent_data.main_data.rollout_ids,
                main_advantages = agent_data.main_advantages
            )
            )
        return advantage_packets
    


    def set_advantage_alignment_data(self, advantage_packets: list[AdvantagePacket]):
        """
        Receive advantage packets from other players.
        These contain the advantages of the other players' rollouts estimated by them.
        """
        assert 2 >= len(advantage_packets) > 0, "At least one advantage packet must be provided."

        for agent_data in self.training_data.values():

            for co_agent_packet in advantage_packets:
                agent_id = agent_data.agent_id
                co_agent_id = co_agent_packet.agent_id
                if agent_id == co_agent_id:
                    continue
                agent_rollout_ids = agent_data.main_data.rollout_ids
                agent_advantages = agent_data.main_advantages
                co_agent_advantages = co_agent_packet.main_advantages
                co_agent_rollout_ids = co_agent_packet.rollout_ids
                B = agent_advantages.shape[0]
                assert agent_advantages.shape[0] == agent_advantages.shape[0], "Batch dimensions must match for advantage alignment."
                # Get co-agent advantages in the right order
                if B > 1:
                    permutation = []
                    for id in agent_rollout_ids: permutation.append(torch.where(id == co_agent_rollout_ids)[0].item())
                    co_agent_advantages = torch.permute(co_agent_advantages, permutation)
                assert torch.all(co_agent_advantages.offsets().diff() == agent_advantages.offsets().diff()), "Number of advantages must match for advantage alignment."
 
                # Get padded tensors (advantage alignment is invariant to padding)
                jagged_lengths = self.batch_advantages.offsets().diff()
                padded_main_advantages = torch.nested.to_padded_tensor(agent_data.main_advantages, padding=0.0)
                padded_alternative_advantages = torch.nested.to_padded_tensor(agent_data.alternative_advantages, padding=0.0)
                padded_co_agent_advantages = torch.nested.to_padded_tensor(co_agent_advantages, padding=0.0)

                # Create training batch data
                credits = get_advantage_alignment_credits(
                    a1 = padded_main_advantages,
                    a1_alternative = padded_alternative_advantages,
                    a2 = padded_co_agent_advantages,
                    beta = self.ad_align_beta,
                    gamma = self.discount_factor,
                    exclude_k_equals_t = self.ad_align_exclude_k_equals_t,
                    use_sign = self.ad_align_use_sign,
                    clipping = self.ad_align_clipping,
                    force_coop_first_step = self.ad_align_force_coop_first_step,
                    tally = self.tally
                )
                advantage_alignment_credits = torch.nested.narrow(credits, dim=1, start=0, length=jagged_lengths, layout=torch.jagged)
                advantage_alignment_credits = get_tokenwise_credits(
                    batch_timesteps = agent_data.main_data.batch_timesteps,
                    batch_credits = advantage_alignment_credits,
                )

                # Set training batch
                agent_data.advantage_alignment_credits = advantage_alignment_credits

    def set_policy_gradient_data(self):
        # Concatenate all agents' data
        self.policy_gradient_data = TrainingBatch(
            rollout_ids = torch.cat([data.main_data.rollout_ids for data in self.training_data.values()]),
            batch_input_ids = torch.nested.nested_tensor([t for data in self.training_data.values() for t in data.main_data.batch_input_ids.unbind()], layout=torch.jagged),
            batch_action_mask = torch.nested.nested_tensor([t for data in self.training_data.values() for t in data.main_data.batch_action_mask.unbind()], layout=torch.jagged),
            batch_credits = torch.nested.nested_tensor([t for data in self.training_data.values() for t in data.advantage_alignment_credits.unbind()], layout=torch.jagged)
        )


