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
from mllm.training.trainer_independent import TrainerNaive
from mllm.training.training_data_utils import *
from mllm.training.training_data_utils import (
    AdvantagePacket,
    TrainingBatch,
    TrajectoryBatch,
    get_tokenwise_credits,
)
from mllm.utils.resource_context import resource_logger_context

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))


class TrainerSumRewards(TrainerNaive):
    def receive_advantage_data(self, advantage_packets: list[AdvantagePacket]):
        """
        Sums the advantages of the other trainers
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
                agent_rollout_ids = agent_data.rollout_ids
                agent_advantages = agent_data.batch_credits
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
                ), "Number of advantages must match in order to sum them up."

                # Get padded tensors (advantage alignment is invariant to padding)
                lengths = torch.tensor(
                    [len(t) for t in agent_advantages],
                    device=self.device,
                    dtype=torch.long,
                )
                padded_main_advantages = pad_sequence(
                    agent_advantages, batch_first=True, padding_value=0.0
                )

                padded_co_agent_advantages = pad_sequence(
                    co_agent_advantages, batch_first=True, padding_value=0.0
                )

                # Create training batch data
                sum_of_ad_credits = padded_main_advantages + padded_co_agent_advantages
                self.rollout_tally.add_metric(path=["sum_of_ad_credits"], rollout_tally_item=RolloutTallyItem(crn_ids=agent_data.main_data.crn_ids, rollout_ids=agent_data.main_data.rollout_ids, agent_ids=agent_data.main_data.agent_ids, metric_matrix=sum_of_ad_credits))

                if not self.skip_discounted_state_visitation:
                    sum_of_ad_credits = get_discounted_state_visitation_credits(
                        sum_of_ad_credits,
                        self.discount_factor,
                    )
                    self.rollout_tally.add_metric(path=["discounted_state_visitation_credits"], rollout_tally_item=RolloutTallyItem(crn_ids=agent_data.main_data.crn_ids, rollout_ids=agent_data.main_data.rollout_ids, agent_ids=agent_data.main_data.agent_ids, metric_matrix=sub_tensors["discounted_state_visitation_credits"]))

                # Slice back to jagged and convert to tokenwise credits
                sum_of_ad_credits = [
                    sum_of_ad_credits[i, : lengths[i]] for i in range(B)
                ]
                self.training_data[agent_id].batch_credits = sum_of_ad_credits
