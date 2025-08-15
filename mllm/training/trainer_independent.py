"""

"""
import logging
import os
import sys
from typing import Union

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from pandas._libs.tslibs.offsets import CBMonthBegin
from peft import LoraConfig
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer

from mllm.markov_games.rollout_tree import *
from mllm.markov_games.rollout_tree import RolloutTreeRootNode
from mllm.training.credit_methods import (
    get_discounted_returns,
    get_discounted_state_visitation_credits,
    get_generalized_advantage_estimates,
    get_rloo_credits,
)
from mllm.training.tally_basic import Tally
from mllm.training.tally_tokenwise import ContextualizedTokenwiseTally
from mllm.training.tokenize_chats import *
from mllm.training.tokenize_chats import process_training_chat
from mllm.training.trainer_common import BaseTrainer
from mllm.training.training_data_utils import *
from mllm.training.training_data_utils import (
    TrainingBatch,
    TrajectoryBatch,
    get_tokenwise_credits,
)
from mllm.utils.resource_context import resource_logger_context

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))


class TrainerNaive(BaseTrainer):
    def set_agent_trajectory_data(
        self, agent_id: str, roots: list[RolloutTreeRootNode]
    ) -> None:
        """
        TOWRITE
        """
        # TODO: append to current batch data instead, else we will only train for one agent!
        self.policy_gradient_data = None

        # Tensorize Chats
        rollout_ids = []
        batch_input_ids = []
        batch_action_mask = []
        batch_timesteps = []
        batch_state_ends_mask = []
        batch_reasoning_limit_tuples = []
        batch_rewards = []
        for root in roots:
            rollout_id = root.id
            self.debug_path_list.append(
                "mgid:" + str(rollout_id) + "_agent_id:" + agent_id
            )
            rollout_ids.append(rollout_id)
            chat, rewards = get_main_chat_list_and_rewards(agent_id=agent_id, root=root)
            (
                input_ids,
                action_mask,
                timesteps,
                state_ends_mask,
                reasoning_limit_tuples,
            ) = process_training_chat(tokenizer=self.tokenizer, chat_history=chat)
            batch_input_ids.append(input_ids)
            batch_action_mask.append(action_mask)
            batch_timesteps.append(timesteps)
            batch_state_ends_mask.append(state_ends_mask)
            batch_rewards.append(rewards)
            batch_reasoning_limit_tuples.append(reasoning_limit_tuples)
        trajectory_batch = TrajectoryBatch(
            rollout_ids=torch.tensor(rollout_ids, dtype=torch.int32),
            batch_input_ids=batch_input_ids,
            batch_action_mask=batch_action_mask,
            batch_timesteps=batch_timesteps,
            batch_state_ends_mask=batch_state_ends_mask,
            batch_rewards=batch_rewards,
            batch_reasoning_limit_tuples=batch_reasoning_limit_tuples,
        )

        # Get Advantages
        batch_advantages: torch.FloatTensor = (
            self.get_advantages_with_critic_gradient_accumulation(trajectory_batch)
        )
        if self.critic_optimizer is not None:
            self.critic_optimizer.step()
            self.critic_optimizer.zero_grad()

        trajectory_batch.batch_credits = batch_advantages

        # Discount state visitation (the mathematically correct way)
        if not self.skip_discounted_state_visitation:
            for i in range(len(batch_advantages)):
                batch_advantages[i] = get_discounted_state_visitation_credits(
                    batch_advantages[i].unsqueeze(0),
                    self.discount_factor,
                ).squeeze(0)

        self.training_data[agent_id] = trajectory_batch

    def receive_advantage_data(self, advantage_packets: list[AdvantagePacket]):
        """
        This trainer ignores the advantages of the other trainers.
        """
        pass

    def share_advantage_data(self) -> list[AdvantagePacket]:
        """
        Share the advantage data with other agents.
        Returns:
            AdvantagePacket: The advantage packet containing the agent's advantages.
        """
        logger.info(f"Sharing advantage data.")
        advantage_packets = []
        for agent_id, agent_data in self.training_data.items():
            advantage_packets.append(
                AdvantagePacket(
                    agent_id=agent_id,
                    rollout_ids=agent_data.rollout_ids,
                    main_advantages=agent_data.batch_credits,
                )
            )
        return advantage_packets
