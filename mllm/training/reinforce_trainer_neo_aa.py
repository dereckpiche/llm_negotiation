from mllm.training.reinforce_trainer import ReinforceTrainerWRS
from dataclasses import dataclass
import numpy as np

# Sent to and fro
RolloutID = str
@dataclass
class advantage_packet:
    rollout_advantages: dict[RolloutID, np.ndarray]

class NeoAATrainer(ReinforceTrainerWRS):
    """
    Extends the reinforce trainer to support Neo Advantage Alignment.
    """

    @staticmethod
    def get_advantage_alignment_weights(
        advantages: np.ndarray,
        beta: float,
        gamma: float,
    ) -> np.ndarray:
        """
        The advantage alignment credit is calculated as

        \[
            A^*(s_t, a_t, b_t) = A^1(s_t, a_t, b_t) + \beta \cdot
            \left( \sum_{k < t} \gamma^{t-k} A^1(s_k, a_k, b_k) \right)
            A^2(s_t, a_t, b_t)
        \]

        Here, the weights are defined as \( \beta \cdot
            \left( \sum_{k < t} \gamma^{t-k} A^1(s_k, a_k, b_k) \)
        """
        # TODO: verify
        # Regular alignment terms
        T = advantages.shape[1]
        discounted_advantages = advantages * (gamma * np.ones(shape=(1, T))) ** (
            -np.arange(0, T, 1)
        )
        # Identity is for \( k < t \), remove for \( k \leq t \)
        discounted_sums_advantages = discounted_advantages @ (
            np.triu(np.ones((T, T))) - np.identity(T)
        )
        t_discounts = (gamma * np.ones(shape=(1, T))) ** (np.arange(0, T, 1))
        adalign_weights = beta * t_discounts * discounted_sums_advantages
        return adalign_weights

    def advantages_to_aa_credits(
        self,
        a1: np.ndarray,
        a2: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate the advantage alignment credits with vectorization, as described in https://arxiv.org/abs/2406.14662.
        Applies normalization, sign, clipping, and regularization as specified in config.
        Optionally logs intermediate and final metrics.

        The advantage alignment credit is calculated as:
        \[
            A^*(s_t, a_t, b_t) = A^1(s_t, a_t, b_t) + \\beta \\gamma \\cdot
            \\left( \\sum_{k < t} \\gamma^{t-k} A^1(s_k, a_k, b_k) \\right)
            A^2(s_t, a_t, b_t)
        \]

        Args:
            a1 (np.ndarray): The first advantage array.
            a2 (np.ndarray): The second advantage array.

        Returns:
            np.ndarray: The advantage alignment terms.
        """
        if len(a1.shape) == 1:
            a1 = a1[None, :]
        if len(a2.shape) == 1:
            a2 = a2[None, :]
        assert a1.shape == a2.shape, "Not the same shape"
        T = a1.shape[1]
        a1 = np.array(a1)
        a2 = np.array(a2)
        gamma = self.config.discount_factor
        beta = self.config.ad_align_beta

        adalign_weights = self.get_advantage_alignment_weights(
            advantages=a1, beta=beta, gamma=gamma
        )

        self.tally.add_metric(
            path=["raw_advantage_alignment_weights"], metric=adalign_weights
        )

        # Use sign
        if self.config.use_sign_in_ad_align:
            assert beta == 1.0, "beta should be 1.0 when using sign"
            positive_signs = adalign_weights > 0
            negative_signs = adalign_weights < 0
            adalign_weights[positive_signs] = 1
            adalign_weights[negative_signs] = -1
            self.tally.add_metric(
                path=["adalign_weights_ratio_positive_signs"],
                metric=positive_signs.sum() / adalign_weights.size,
            )
            self.tally.add_metric(
                path=["adalign_weights_ratio_negative_signs"],
                metric=negative_signs.sum() / adalign_weights.size,
            )
            # (rest are 0)

            self.tally.add_metric(
                path=["ad_align_weights_after_using_sign"], metric=adalign_weights
            )

        # Use clipping
        if self.config.ad_align_clipping not in [0.0, None]:

            upper_mask = adalign_weights > 1
            lower_mask = adalign_weights < -1

            adalign_weights = np.clip(
                adalign_weights,
                -self.config.ad_align_clipping,
                self.config.ad_align_clipping,
            )
            clipping_ratio = (np.sum(upper_mask) + np.sum(lower_mask)) / upper_mask.size

            self.tally.add_metric(
                path=["ad_align_clipping_ratio"], metric=clipping_ratio
            )

            self.tally.add_metric(
                path=["ad_align_weights_after_clipping"], metric=adalign_weights
            )

        # 1/1+t Regularization
        if self.config.use_time_regularization_in_ad_align:
            t_values = np.arange(1, T + 1)
            adalign_weights = adalign_weights / t_values
            self.tally.add_metric(
                path=["adalign_weights_after_1_over_t_reg"], metric=adalign_weights
            )

        # Use coop on t=0
        if self.config.ad_align_force_coop_first_step:
            adalign_weights[:, 0] = 1
            self.tally.add_metric(
                path=["adalign_weights_after_force_coop_first_step"],
                metric=adalign_weights,
            )

        opp_shaping_terms = adalign_weights * a2

        self.tally.add_metric(
            path=["ad_align_opp_shaping_terms"], metric=opp_shaping_terms
        )

        # Normalize alignment terms (across same time step)
        if self.config.use_variance_regularization_in_ad_align:
            # TODO: verify
            reg_coef = np.std(a1[:, -1]) / (np.std(opp_shaping_terms[:, -1]) + 1e-9)
            opp_shaping_terms *= reg_coef
            self.tally.add_metric(
                path=["opp_shaping_terms_after_var_reg"], metric=opp_shaping_terms
            )

        ad_align_credits = a1 + opp_shaping_terms

        self.tally.add_metric(
            path=["final_advantage_alignment_credits"], metric=ad_align_credits
        )

        self.logger.info(f"\n \n After AdAlign \n  {ram_usage()} \n {vram_usage()}")

        return ad_align_credits.squeeze()

    def send_advantages(self) -> dict:
        """
        Prepares and returns reward shaping information (game IDs, step rewards, step credits) to be shared with opponent trainers.

        Returns:
            dict: Dictionary containing game IDs, step rewards, and step credits.
        """

        @dataclass

        return info

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
