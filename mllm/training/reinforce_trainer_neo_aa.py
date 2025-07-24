from mllm.training.reinforce_trainer import ReinforceTrainerWRS
from dataclasses import dataclass
import numpy as np

@dataclass
class NeoAdlignData:
    trajectory : TensorTrajectory
    alternative_action_branches: dict[timestep, list[TensorTrajectory]]

# Sent to and fro
RolloutID = str
@dataclass
class advantage_packet:
    rollout_advantages: dict[RolloutID, np.ndarray]

class NeoAATrainer(ReinforceTrainerWRS):
    """
    Extends the reinforce trainer to support Neo Advantage Alignment.
    """
    def send_advantages(self) -> dict[markov_game_Id, torch.Float]:
        """
        Prepares and returns reward shaping information (game IDs, step rewards, step credits) to be shared with opponent trainers.

        Returns:
            dict: Dictionary containing game IDs, step rewards, and step credits.
        """

        @dataclass

        return info

    def send_info()
        return self.send_advantages)_

    def receive_advantages(self, : dict) -> None:
        """
        Incorporates reward shaping information received from opponent trainers.
        Aligns opponent data with local data, and applies sum or advantage alignment as specified in config.
        This is a bit convoluted, but it works with self play.

        Args:
            co_trainer_info (dict): Dictionary containing opponent trainer info.
        """

        # Map each id's to integers
        intmap = {
            s: i
            for i, s in enumerate(
                set(
                    self.agent_ids
                    + self.game_ids
                    + co_trainer_info.get("game_ids")
                    + co_trainer_info.get("agent_ids")
                )
            )
        }

        def intmapf(ar):
            return np.array([intmap[a] for a in ar])

        agent_ids = intmapf(self.agent_ids)
        game_ids = intmapf(self.game_ids)
        co_trainer_agent_ids = intmapf(co_trainer_info.get("agent_ids"))
        co_trainer_game_ids = intmapf(co_trainer_info.get("game_ids"))
        B = len(self.step_credits)

        op_step_credits = []  # Get credits of other agent in right order
        for i in range(B):
            idx = (co_trainer_game_ids == game_ids[i]) & (
                co_trainer_agent_ids != agent_ids[i]
            )
            idx = np.where(idx)[0].item()
            op_step_credits.append(co_trainer_info.get("step_credits")[idx])

        if self.config.use_sum_credits:
            for i in range(len(self.step_credits)):
                self.tally.add_metric(
                    path=["credits_before_sum_credits"], metric=self.step_credits[i]
                )
                self.step_credits[i] += op_step_credits[i]
                self.tally.add_metric(
                    path=["credits_after_sum_credits"], metric=self.step_credits[i]
                )

        if self.config.use_advantage_alignment:
            for i in range(B):
                self.step_credits[i] = self.advantages_to_aa_credits(
                    a1=self.step_credits[i][None, :], a2=op_step_credits[i][None, :]
                )
