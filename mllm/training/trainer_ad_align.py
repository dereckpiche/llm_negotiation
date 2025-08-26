import copy
import logging
import sys
from dataclasses import dataclass
from typing import Tuple

import torch
from torch.nn.utils.rnn import pad_sequence

from mllm.markov_games.rollout_tree import (
    ChatTurn,
    RolloutTreeBranchNode,
    RolloutTreeRootNode,
)
from mllm.training.credit_methods import (
    get_advantage_alignment_credits,
    get_discounted_state_visitation_credits,
)
from mllm.training.tally_basic import Tally
from mllm.training.tally_tokenwise import ContextualizedTokenwiseTally
from mllm.training.tokenize_chats import process_training_chat
from mllm.training.trainer_common import BaseTrainer
from mllm.training.training_data_utils import (
    AdvantagePacket,
    TrainingBatch,
    TrainingChatTurn,
    TrajectoryBatch,
    get_main_chat_list_and_rewards,
    get_tokenwise_credits,
)
from mllm.utils.resource_context import resource_logger_context

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))

RolloutId = int
AgentId = str


@dataclass
class AdAlignTrainingData:
    agent_id: str
    main_data: TrajectoryBatch
    # list-of-tensors: per rollout advantages with length jT
    main_advantages: list[torch.FloatTensor] | None = None
    # list-of-tensors: per rollout matrix (jT, A)
    alternative_advantages: list[torch.FloatTensor] | None = None
    advantage_alignment_credits: list[torch.FloatTensor] | None = None


def get_alternative_chat_histories(
    agent_id: str, root: RolloutTreeRootNode
) -> list[list[TrainingChatTurn], list[torch.FloatTensor]]:
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
        assert isinstance(
            current_node, RolloutTreeBranchNode
        ), "Current node should be a branch node."
        main_node = current_node.main_child
        branches = current_node.branches
        current_node = main_node.child

        # Get the `A` alternative trajectories
        alternative_nodes = branches[agent_id]
        for alt_node in alternative_nodes:
            post_branch_chat, post_branch_rewards = get_main_chat_list_and_rewards(
                agent_id=agent_id, root=alt_node
            )
            branch_chat = pre_branch_chat + post_branch_chat
            alternative_chats.append(branch_chat)
            alternative_rewards.append(
                torch.cat([torch.tensor(pre_branch_rewards), post_branch_rewards])
            )

        chat_turns: list[ChatTurn] = main_node.step_log.action_logs[agent_id].chat_turns
        chat_turns: list[TrainingChatTurn] = [
            TrainingChatTurn(time_step=main_node.time_step, **turn.model_dump())
            for turn in chat_turns
        ]

        pre_branch_chat.extend(chat_turns)
        pre_branch_rewards.append(
            main_node.step_log.simulation_step_log.rewards[agent_id]
        )

    return alternative_chats, alternative_rewards


class TrainerAdAlign(BaseTrainer):
    """
    Extends the reinforce trainer to support Advantage Alignment.
    """

    def __init__(
        self,
        ad_align_beta: float,
        ad_align_gamma: float,
        ad_align_exclude_k_equals_t: bool,
        ad_align_use_sign: bool,
        ad_align_clipping: float,
        ad_align_force_coop_first_step: bool,
        use_old_ad_align: bool,
        use_time_regularization: bool,
        rloo_branch: bool,
        reuse_baseline: bool,
        *args,
        **kwargs,
    ):
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
        self.use_old_ad_align = use_old_ad_align
        self.use_time_regularization = use_time_regularization
        self.rloo_branch = rloo_branch
        self.reuse_baseline = reuse_baseline
        self.training_data: dict[AgentId, AdAlignTrainingData] = {}
        self.debug_path_list: list[str] = []

    def set_agent_trajectory_data(
        self, agent_id: str, roots: list[RolloutTreeRootNode]
    ):
        """
        TOWRITE
        Set the advantage alignment data for the trainer.
        """

        B = len(roots)  # Number of rollouts

        # For main rollouts
        batch_rollout_ids = []
        batch_rng_seeds = []
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

        try:
            A = len(roots[0].child.branches[agent_id])  # Number of alternative actions
        except:
            A = 0

        for root in roots:
            rollout_id = root.id
            self.debug_path_list.append(
                "mgid:" + str(rollout_id) + "_agent_id:" + agent_id
            )
            # Get main trajectory
            batch_rollout_ids.append(rollout_id)
            batch_rng_seeds.append(root.rng_seed)
            main_chat, main_rewards = get_main_chat_list_and_rewards(
                agent_id=agent_id, root=root
            )
            (
                input_ids,
                action_mask,
                timesteps,
                state_ends_mask,
            ) = process_training_chat(tokenizer=self.tokenizer, chat_history=main_chat)
            batch_input_ids.append(input_ids)
            batch_action_mask.append(action_mask)
            batch_timesteps.append(timesteps)
            batch_state_ends_mask.append(state_ends_mask)
            batch_rewards.append(main_rewards)
            jT = main_rewards.numel()  # TODO: better than this
            jT_list.append(jT)
            if A > 0:
                # We get the branching time steps for each of the `jT` time steps in the main trajectory.
                branching_time_steps = [bt for item in range(jT) for bt in A * [item]]
                batch_branching_time_steps.extend(branching_time_steps)

                # Get all of the (jT*A) alternative trajectories in the tree
                # (jT is the number of time steps in the main trajectory, A is the number of alternative actions)
                alternative_chats, alternative_rewards = get_alternative_chat_histories(
                    agent_id=agent_id, root=root
                )
                assert (
                    len(alternative_chats) == A * jT
                ), "Incorrect number of alternative trajectories."

                for chat, rewards in zip(alternative_chats, alternative_rewards):
                    (
                        input_ids,
                        action_mask,
                        timesteps,
                        state_ends_mask,
                    ) = process_training_chat(
                        tokenizer=self.tokenizer, chat_history=chat
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

        trajectory_batch = TrajectoryBatch(
            rollout_ids=torch.tensor(batch_rollout_ids, dtype=torch.int32),  # (B,)
            rng_seeds=torch.tensor(batch_rng_seeds, dtype=torch.int32),
            batch_input_ids=batch_input_ids,
            batch_action_mask=batch_action_mask,
            batch_timesteps=batch_timesteps,
            batch_state_ends_mask=batch_state_ends_mask,
            batch_rewards=batch_rewards,
        )
        # Get Advantages & Train Critic
        with resource_logger_context(
            logger, "Get advantages with critic gradient accumulation"
        ):
            self.batch_advantages: torch.FloatTensor = (
                self.get_advantages_with_critic_gradient_accumulation(trajectory_batch)
            )  # (B, jT)

        if A > 0:
            # Here, `A` is the number of alternative actions / trajectories taken at each time step.
            # For each of the `B` rollout perspectives, at each of its jT (`j` is for jagged, since each main rollout may be of a different length) steps, we take A alternate trajectories (from different actions).
            # Therefore, we have ∑jT * A trajectories to process. If each of the main trajectories have T steps, we will have `B*T*A` to process.
            with resource_logger_context(logger, "Create alternative trajectory batch"):
                sum_jT = int(torch.sum(jT_list).item())
                jT_list = (
                    jT_list.int().tolist()
                )  # (jT,) # (we only want the advantages where we branched out)
                alternative_trajectory_batch = TrajectoryBatch(
                    rollout_ids=torch.zeros(A * sum_jT, dtype=torch.int32),
                    rng_seeds=torch.zeros(A * sum_jT, dtype=torch.int32),
                    batch_input_ids=alternative_batch_input_ids,
                    batch_action_mask=alternative_batch_action_mask,
                    batch_timesteps=alternative_batch_timesteps,
                    batch_state_ends_mask=alternative_batch_state_ends_mask,
                    batch_rewards=alternative_batch_rewards,
                )

            # Get alternative advantages
            # BAAs stands for batch alternative advantages
            # (torch nested tensors have very little api support, so we have to do some odd manual work here)
            with resource_logger_context(
                logger, "Compute alternative advantage estimates"
            ):
                BAAs_list = self.get_advantages_with_critic_gradient_accumulation(
                    alternative_trajectory_batch
                )  # list length (∑jT * A), each (jT',)
                # Pad alternative advantages to (∑jT*A, P)

                BAAs_padded = pad_sequence(
                    BAAs_list, batch_first=True, padding_value=0.0
                )
                branch_idx = torch.tensor(
                    batch_branching_time_steps,
                    device=BAAs_padded.device,
                    dtype=torch.long,
                )
                gathered = BAAs_padded.gather(
                    dim=1, index=branch_idx.unsqueeze(1)
                ).squeeze(1)
                # Reshape and split per rollout, then transpose to (jT_i, A)
                gathered = gathered.view(A, sum_jT)  # (A, ∑jT)
                blocks = list(
                    torch.split(gathered, jT_list, dim=1)
                )  # len B, shapes (A, jT_i)
                BAAs = [
                    blk.transpose(0, 1).contiguous() for blk in blocks
                ]  # list of (jT_i, A)

        self.training_data[agent_id] = AdAlignTrainingData(
            agent_id=agent_id,
            main_data=trajectory_batch,
            main_advantages=self.batch_advantages,
            alternative_advantages=BAAs if A > 0 else None,
        )

    def share_advantage_data(self) -> list[AdvantagePacket]:
        """
        Share the advantage alignment data with other agents.
        Returns:
            AdvantagePacket: The advantage packet containing the agent's advantages.
        """
        logger.info(f"Sharing advantage alignment data.")
        advantage_packets = []
        for _, agent_data in self.training_data.items():
            advantage_packets.append(
                AdvantagePacket(
                    agent_id=agent_data.agent_id,
                    rollout_ids=agent_data.main_data.rollout_ids,
                    main_advantages=agent_data.main_advantages,
                )
            )
        return advantage_packets

    def receive_advantage_data(self, advantage_packets: list[AdvantagePacket]):
        """
        Receive advantage packets from other players.
        These contain the advantages of the other players' rollouts estimated by them.
        """
        logger.info(f"Receiving advantage packets.")

        assert (
            2 >= len(advantage_packets) > 0
        ), "At least one advantage packet must be provided."

        for agent_id, agent_data in self.training_data.items():
            for co_agent_packet in advantage_packets:
                co_agent_id = co_agent_packet.agent_id
                if agent_id == co_agent_id:
                    continue
                agent_rollout_ids = agent_data.main_data.rollout_ids
                agent_advantages = agent_data.main_advantages
                co_agent_advantages = co_agent_packet.main_advantages
                co_agent_rollout_ids = co_agent_packet.rollout_ids
                B = len(agent_advantages)
                # Get co-agent advantages in the right order
                permutation = []
                for id in agent_rollout_ids:
                    permutation.append(
                        torch.where(id == co_agent_rollout_ids)[0].item()
                    )
                co_agent_advantages = [co_agent_advantages[i] for i in permutation]
                assert all(
                    a.shape[0] == b.shape[0]
                    for a, b in zip(co_agent_advantages, agent_advantages)
                ), "Number of advantages must match for advantage alignment."

                # Get padded tensors (advantage alignment is invariant to padding)
                lengths = torch.tensor(
                    [len(t) for t in agent_advantages],
                    device=self.device,
                    dtype=torch.long,
                )
                padded_main_advantages = pad_sequence(
                    agent_advantages, batch_first=True, padding_value=0.0
                )
                if agent_data.alternative_advantages:
                    padded_alternative_advantages = pad_sequence(
                        agent_data.alternative_advantages,
                        batch_first=True,
                        padding_value=0.0,
                    )  # (B, P, A)
                else:
                    padded_alternative_advantages = None
                padded_co_agent_advantages = pad_sequence(
                    co_agent_advantages, batch_first=True, padding_value=0.0
                )

                # Create training batch data
                credits = get_advantage_alignment_credits(
                    a1=padded_main_advantages,
                    a1_alternative=padded_alternative_advantages,
                    a2=padded_co_agent_advantages,
                    beta=self.ad_align_beta,
                    gamma=self.discount_factor,
                    exclude_k_equals_t=self.ad_align_exclude_k_equals_t,
                    use_sign=self.ad_align_use_sign,
                    clipping=self.ad_align_clipping,
                    force_coop_first_step=self.ad_align_force_coop_first_step,
                    use_old_ad_align=self.use_old_ad_align,
                    use_time_regularization=self.use_time_regularization,
                    rloo_branch=self.rloo_branch,
                    reuse_baseline=self.reuse_baseline,
                    tally=self.tally,
                )

                if not self.skip_discounted_state_visitation:
                    credits = get_discounted_state_visitation_credits(
                        credits,
                        self.discount_factor,
                    )

                # Slice back to jagged
                advantage_alignment_credits = [
                    credits[i, : lengths[i]] for i in range(B)
                ]
                # Replace stored training data for this agent by the concrete trajectory batch
                # and attach the computed credits for policy gradient.
                self.training_data[agent_id] = agent_data.main_data
                self.training_data[agent_id].batch_credits = advantage_alignment_credits
