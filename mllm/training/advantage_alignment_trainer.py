from mllm.training.reinforce_trainer import ReinforceTrainerWRS
from dataclasses import dataclass
import torch
import numpy as np
from typing import Tuple

@dataclass
class NeoAdlignData:
    trajectory : TensorTrajectory
    alternative_action_branches: dict[timestep, list[TensorTrajectory]]

class AdvantageCollection:
    # B := batch size, S := number of tokens / seq. length, A := number of alternative actions taken. `j` stands for jagged (see pytorch nested tensors.)
    rollout_ids: torch.IntTensor # (B,)
    main_advantages: torch.FloatTensor # (B, jS)
    alternative_advantages: torch.FloatTensor # (B, A, jS)
    """
    """

# Sent to and fro
RolloutID = str
MarkovGameID = str
AgentID = str

@dataclass
class advantage_packet:
    agent_id: AgentID
    advantage_collection: AdvantageCollection



class AdAlignTrainer(BaseTrainer):
    """
    Extends the reinforce trainer to support Neo Advantage Alignment.
    """

    def set_policy_gradient_data(self, roots: list[RolloutTreeRootNode], coordinator):

        # ---------------------------------------------------
        # Receive advantage estimations from other players
        # ---------------------------------------------------

        co_agent_advantages : AdvantageCollection = None # TODO
        assert self.advantages.main_advantages.shape[0] == self.co_agent_advantages.main_advantages.shape[0], "Advantage shapes must match!"

        # Get co-agent advantages in the right order
        permutation = []
        for id in self.advantages.rollout_ids: permutation.append(torch.where(id == co_agent_advantages.rollout_ids)[0].item())
        self.coagent_main_advantages = torch.permute(co_agent_advantages.rollout_ids, permutation)

        # ---------------------------------------------------
        # Create final training data
        # ---------------------------------------------------

        # Create training batch data
        credits = get_advantage_alignment_credits(
            a1 = self.advantages.main_advantages,
            a1_star = self.advantages.alternative_advantages,
            a2 = self.coagent_main_advantages
        )

        # Set training batch
        self.training_batch = TrainingBatch(
            rollout_ids = self.main_trajectories.rollout_ids,
            batch_input_ids = self.main_trajectories.batch_input_ids,
            batch_action_mask = self.main_trajectories.batch_action_mask,
            batch_credits = credits
        )




class AdAlignCoordinator:
    def __init__(
        self,
        *trainers: Tuple[AdAlignTrainer, ...]
    ):
        self.trainers = trainers

    def train(self):
        advantage_pool = []

        # Get Advantages
        for trainer in self.trainers:
            advantage_pool.extend[trainer.estimate_advantages()]

        # Share Advantages & Train
        for trainer in self.trainers:
            trainer.receive_advantages(advantage_pool)
            trainer.train()
