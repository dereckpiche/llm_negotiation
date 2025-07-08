import logging
import os
import random
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from copy import deepcopy
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from mllm.training.reinforce_trainer_config import RtConfig
from mllm.training.reinforce_trainer_tally import RtTally
from mllm.training.process_training_chat import process_training_chat
from mllm.utils.common_imports import *
from mllm.utils.time_and_memory_utils import *
from mllm.utils.print_logger import *
from typing import Union


class ReinforceTrainerWRS:
    """
    REINFORCE trainer with reward shaping.
    To be used in a multi-agent context.
    Generalizes to single-agent case if used carefully.
    """

    # TODO: use in place operations to minimize GPU usage // allocation
    # TODO: include torch.compile if use perform multiple gradient steps on same data
    # TODO: add GAE
    # TODO: token end_ids for different model classes
    # TODO: Add option for different discounted normalization scheme
    # TODO: check AdAlign code for normalization
    # TODO: add value function
    # TODO: log top k probs
    # TODO: add lr scheduler support

    # so that we can check the gradient magnitude of the different relative terms (in percentages)

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        critic: Union[AutoModelForCausalLM, None],
        critic_optimizer: Union[torch.optim.Optimizer, None],
        critic_lr_scheduler: Union[torch.optim.lr_scheduler.LRScheduler, None],
        config: RtConfig,
        save_path: str,
    ):
        """
        Initialize the REINFORCE trainer with reward shaping for multi-agent or single-agent training.

        Args:
            model (AutoModelForCausalLM): The main policy model.
            tokenizer (AutoTokenizer): Tokenizer for the model.
            optimizer (torch.optim.Optimizer): Optimizer for the policy model.
            lr_scheduler (torch.optim.lr_scheduler.LRScheduler): Learning rate scheduler for the policy model.
            critic (AutoModelForCausalLM or None): Critic model for value estimation (optional).
            critic_optimizer (torch.optim.Optimizer or None): Optimizer for the critic model (optional).
            critic_lr_scheduler (torch.optim.lr_scheduler.LRScheduler or None): LR scheduler for the critic (optional).
            config (RtConfig): Configuration object for training.
        """
        # TODO: add lr schedulers

        model.train()
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left"  # needed for flash attention
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.config = config
        self.accelerator = Accelerator()
        self.model, self.optimizer, self.critic, self.critic_optimizer = (
            self.accelerator.prepare(model, optimizer, critic, critic_optimizer)
        )

        self.critic_lr_scheduler = critic_lr_scheduler

        self.tally = RtTally(tokenizer=tokenizer)

        self.logger = PrintLogger(logging.getLogger("reinforcer_trainer_logger"))

        if self.config.use_gradient_checkpointing == True:
            self.logger.info("Enabling gradient checkpointing.")
            self.model.gradient_checkpointing_enable(dict(use_reentrant=False))
            self.critic.gradient_checkpointing_enable(dict(use_reentrant=False))

        self.save_path = save_path

        # Load states if already exist
        self.policy_optimizer_path = os.path.join(
            self.save_path, "policy_optimizer_state.pt"
        )
        if os.path.exists(self.policy_optimizer_path):
            self.logger.info(
                f"Loading policy optimizer state from {self.policy_optimizer_path}"
            )
            self.optimizer.load_state_dict(torch.load(self.policy_optimizer_path))

        self.critic_optimizer_path = os.path.join(
            self.save_path, "critic_optimizer_state.pt"
        )
        if (
            os.path.exists(self.critic_optimizer_path)
            and self.critic_optimizer is not None
        ):
            self.logger.info(
                f"Loading critic optimizer state from {self.critic_optimizer_path}"
            )
            self.critic_optimizer.load_state_dict(
                torch.load(self.critic_optimizer_path)
            )

        # TODO
        # log data type of model
        # log adapter type, rank, etc.
        # log optimizer learning rate
        # log model data type
        # log adapter data type

    def mask_non_restricted_token_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Masks logits so that only allowed tokens (as specified in config.restrict_tokens)
        and the EOS token are active.
        All other logits are set to -inf, effectively removing them from the softmax.

        Args:
            logits (torch.Tensor): The logits tensor of shape (B, S, V).

        Returns:
            torch.Tensor: The masked logits tensor.
        """
        # TODO: verify. Not sure what we do here is differentiable
        # also, we recompute for nothing

        if self.config.restrict_tokens is not None:
            allowed_token_ids = []
            for token in self.config.restrict_tokens:
                token_ids = self.tokenizer(token, add_special_tokens=False)["input_ids"]
                allowed_token_ids.append(token_ids[0])
            allowed_token_ids.append(
                self.tokenizer.eos_token_id
            )  # This token should always be active
            allowed_token_ids = torch.tensor(allowed_token_ids, device=logits.device)
            # Mask log_probs and probs to only allowed tokens
            mask = torch.zeros_like(logits).bool()  # (B, S, V)
            mask[..., allowed_token_ids] = True
            logits = torch.where(
                mask,
                logits,
                torch.tensor(-float("inf"), device=logits.device),
            )

        return logits

    def get_expected_entropy(
        self,
        paths: list[str],
        contexts: torch.Tensor,
        shifted_contexts: torch.Tensor,
        logits: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the expected entropy of the action distribution for LLM-generated token sequences.
        Optionally logs contextualized entropy metrics if enabled in config.

        Args:
            game_ids (list[str]): List of game IDs for each sample in the batch.
            contexts (torch.Tensor): Input context tensor.
            shifted_contexts (torch.Tensor): Shifted context tensor (for next-token prediction).
            logits (torch.Tensor): Logits tensor from the model.
            action_mask (torch.Tensor): Mask indicating which tokens are actions.

        Returns:
            torch.Tensor: The expected entropy (scalar).
        """
        try:
            B, S, T = logits.shape
        except:
            print("Missing batch dimension.")

        token_entropy_terms = -F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)
        # (B, S, T)

        # We only take the entropy of actions
        token_entropy_terms *= action_mask[:, :, None]

        if self.config.log_ctz_entropy:
            self.tally.add_contextualized_token_metrics(
                game_ids=paths,
                metric_id="entropy",
                contexts=shifted_contexts,
                metrics=token_entropy_terms.sum(dim=-1),
                action_mask=action_mask,
            )
        expected_entropy = token_entropy_terms.sum()

        del token_entropy_terms

        return expected_entropy

    def get_kl_divergence(
        self,
        game_ids: list[str],
        contexts: torch.Tensor,
        shifted_contexts: torch.Tensor,
        attention_mask: torch.Tensor,
        action_log_probs: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the approximate KL divergence between the current policy
        and a reference (base) model for the action tokens.
        Optionally logs contextualized KL metrics if enabled in config.

        Args:
            game_ids (list[str]): List of game IDs for each sample in the batch.
            contexts (torch.Tensor): Input context tensor.
            shifted_contexts (torch.Tensor): Shifted context tensor (for next-token prediction).
            attention_mask (torch.Tensor): Attention mask for the input.
            action_log_probs (torch.Tensor): Log probabilities of actions taken.
            action_mask (torch.Tensor): Mask indicating which tokens are actions.

        Returns:
            torch.Tensor: The KL divergence (scalar).
        # Ref 1: http://joschu.net/blog/kl-approx.html
        # Ref 2: https://github.dev/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py#L1332
        """

        # Disable policy adapter to run inference on base model
        with torch.no_grad():
            with self.model.disable_adapter():
                ref_model_logits = self.model(
                    input_ids=contexts, attention_mask=attention_mask
                )[0]

        ref_model_logits = ref_model_logits / self.config.temperature

        # (B, S, V)
        ref_model_logits = self.mask_non_restricted_token_logits(
            logits=ref_model_logits
        )
        # (B, S, V)
        ref_model_log_probs = F.log_softmax(ref_model_logits, dim=-1)
        # (B, S, V)
        ref_model_action_log_probs = ref_model_log_probs.gather(
            dim=-1, index=shifted_contexts.unsqueeze(-1)
        ).squeeze(
            -1
        )  # (B,S)
        # Approximating KL Divergence (see refs in docstring)
        kl_div = (
            torch.exp(ref_model_action_log_probs - action_log_probs)
            - (ref_model_action_log_probs - action_log_probs)
            - 1
        )

        if self.config.log_ctz_kl:
            self.tally.add_contextualized_token_metrics(
                game_ids=game_ids,
                metric_id="kl",
                contexts=shifted_contexts,
                metrics=kl_div,
                action_mask=action_mask,
            )

        # We only care about KLD of action tokens
        kl_div *= action_mask

        kl_div = kl_div.sum()

        return kl_div

    def get_gradient_magnitude(self, loss_term: torch.Tensor) -> float:
        """
        Computes the L2 norm of the gradients of the given loss term with respect to the model parameters.

        Args:
            loss_term (torch.Tensor): The loss tensor to compute gradients for.

        Returns:
            float: The L2 norm of the gradients, or 0.0 if no gradients are present.
        """
        with torch.no_grad():
            grads = torch.autograd.grad(
                loss_term,
                [p for p in self.model.parameters() if p.requires_grad],
                retain_graph=True,
                allow_unused=True,
            )
            grads = [g for g in grads if g is not None]
            if not grads:
                return torch.tensor(0.0, device=loss_term.device)
            return torch.norm(torch.stack([g.norm(2) for g in grads])).item()

    def apply_reinforce_step(
        self,
        paths: list[str],
        contexts: list[torch.Tensor],
        credits: list[torch.Tensor],
        action_masks: list[torch.Tensor],
    ) -> None:
        """
        Applies a single REINFORCE policy gradient step using the provided batch of rollouts.
        Handles batching, loss computation (including entropy and KL regularization), gradient accumulation, and optimizer step.
        Optionally logs various metrics and statistics.

        Args:
            paths (list[str]): List of game complete file paths for each rollout.
            contexts (list[torch.Tensor]): List of context tensors for each rollout.
            credits (list[torch.Tensor]): List of credit tensors (rewards/advantages) for each rollout.
            action_masks (list[torch.Tensor]): List of action mask tensors for each rollout.
        """

        self.logger.info(
            f"\n Before Reinforce Step \n  {ram_usage()} \n {vram_usage()}"
        )

        self.model.train()
        loss = 0.0
        mb_size = self.config.mini_batch_size
        device = self.accelerator.device
        self.tokenizer.padding_side = "left"

        nb_rollouts = len(contexts)
        self.tally.add_metric(path=["nb_rollouts"], metric=nb_rollouts)

        # Count total number of tokens trained on
        total_nb_action_tokens = 0
        for am in action_masks:
            nb_tokens = am.shape[0]
            self.tally.add_metric(path=["nb_tokens", "action_tokens"], metric=nb_tokens)
            total_nb_action_tokens += nb_tokens

        self.tally.add_metric(
            path=["nb_tokens", "batch_action_tokens"], metric=total_nb_action_tokens
        )

        for mb in range(0, len(contexts), mb_size):
            paths_mb = paths[mb : mb + mb_size]

            # Convert sequences to padded tensor minibatches
            tokens_mb = contexts[mb : mb + mb_size]
            tok_out = self.tokenizer.pad(
                {"input_ids": tokens_mb}, padding="longest", return_tensors="pt"
            )
            tokens_mb = tok_out.input_ids.to(device)
            attention_mask = tok_out.attention_mask.to(device)[:, :-1]
            contexts_mb = tokens_mb[:, :-1]
            shifted_contexts_mb = tokens_mb[:, 1:]

            credits_mb = [s[1:] for s in credits[mb : mb + mb_size]]
            credits_mb = (
                pad_sequence(
                    sequences=credits_mb,
                    padding_value=0.0,
                    batch_first=True,
                    padding_side="left",
                )
                .float()
                .to(device)
            )

            action_mask_mb = [am[1:] for am in action_masks[mb : mb + mb_size]]
            action_mask_mb = (
                pad_sequence(
                    sequences=action_mask_mb,
                    padding_value=0.0,
                    batch_first=True,
                    padding_side="left",
                )
                .float()
                .to(device)
            )

            if self.config.log_ctz_next_token_credit:
                self.tally.add_contextualized_token_metrics(
                    paths=paths_mb,
                    metric_id="next_token_credit",
                    contexts=shifted_contexts_mb,
                    metrics=credits_mb,
                    action_mask=action_mask_mb,
                )

            # Forward pass + cast to FP-32 for higher prec.
            logits = self.model(input_ids=contexts_mb, attention_mask=attention_mask)[
                0
            ]  # (B, S, V)

            # Mask non-restricted tokens
            if self.config.restrict_tokens is not None:
                logits = self.mask_non_restricted_token_logits(logits)

            logits /= self.config.temperature  # (B, S, V)

            # Compute new log probabilities
            log_probs = F.log_softmax(logits, dim=-1)  # (B, S, V)

            # Get log probabilities of actions taken during rollouts
            action_log_probs = log_probs.gather(
                dim=-1, index=shifted_contexts_mb.unsqueeze(-1)
            ).squeeze(
                -1
            )  # (B, S)

            if self.config.log_ctz_next_token_log_prob:
                self.tally.add_contextualized_token_metrics(
                    paths=paths_mb,
                    metric_id="next_token_log_prob",
                    contexts=shifted_contexts_mb,
                    metrics=action_log_probs,
                    action_mask=action_mask_mb,
                )

            if self.config.log_ctz_next_token_prob:
                self.tally.add_contextualized_token_metrics(
                    paths=paths_mb,
                    metric_id="next_token_prob",
                    contexts=shifted_contexts_mb,
                    metrics=torch.exp(action_log_probs),
                    action_mask=action_mask_mb,
                )

            if self.config.log_ctz_top_k != 0:
                # Log top K ids and probs

                self.logger.info(
                    f"\n Before Logging Top K \n  {ram_usage()} \n {vram_usage()}"
                )

                top_k_indices = torch.topk(
                    logits, k=self.config.log_ctz_top_k, dim=-1
                ).indices

                if self.config.log_ctz_top_k_tids:
                    self.tally.add_contextualized_token_metrics(
                        paths=paths_mb,
                        metric_id=f"top_{self.config.log_ctz_top_k}_tids",
                        contexts=shifted_contexts_mb,
                        metrics=top_k_indices,
                        action_mask=action_mask_mb,
                        to_tids=True,
                    )
                if self.config.log_ctz_top_k_probs:
                    self.tally.add_contextualized_token_metrics(
                        paths=paths_mb,
                        metric_id=f"top_{self.config.log_ctz_top_k}_probs",
                        contexts=shifted_contexts_mb,
                        metrics=torch.exp(log_probs).gather(
                            dim=-1, index=top_k_indices
                        ),
                        action_mask=action_mask_mb,
                    )

                self.logger.info(
                    f"\n After Logging Top K \n  {ram_usage()} \n {vram_usage()}"
                )

            rewarded_action_log_probs = action_mask_mb * credits_mb * action_log_probs
            # (B, S)
            if self.config.log_ctz_top_clogπ:
                self.tally.add_contextualized_token_metrics(
                    paths=paths_mb,
                    metric_id="next_token_clogπ",
                    contexts=shifted_contexts_mb,
                    metrics=rewarded_action_log_probs,
                    action_mask=action_mask_mb,
                )

            # Add value term to loss
            nb_act_tokens = torch.sum(action_mask_mb)
            mb_value = -rewarded_action_log_probs.sum()
            self.tally.add_metric(
                path=["loss_mb_total", "value_mb_total"], metric=mb_value.item()
            )

            if self.config.log_value_gradient_terms:
                self.tally.add_metric(
                    path=["gradient_term_magnitudes", "value"],
                    metric=self.get_gradient_magnitude(loss_term=mb_value),
                )
            loss += mb_value

            # Add entropy regularization term to loss
            if self.config.entropy_coeff != 0.0:

                self.logger.info(
                    f"\n Before Computing Entropy \n  {ram_usage()} \n {vram_usage()}"
                )

                mb_entropy = self.get_expected_entropy(
                    paths=paths_mb,
                    contexts=contexts_mb,
                    shifted_contexts=shifted_contexts_mb,
                    logits=logits,
                    action_mask=action_mask_mb,
                )
                mb_entropy *= self.config.entropy_coeff
                self.tally.add_metric(
                    path=["loss_mb_total", "entropy_mb_total"], metric=mb_entropy.item()
                )

                if self.config.log_entropy_gradient_terms:
                    self.tally.add_metric(
                        path=["gradient_term_magnitudes", "entropy"],
                        metric=self.get_gradient_magnitude(loss_term=mb_entropy),
                    )
                loss += mb_entropy

                self.logger.info(
                    f"\n After Computing Entropy \n  {ram_usage()} \n {vram_usage()}"
                )

            # Add KL-divergence regularization term to loss
            if self.config.kl_coeff != 0.0:

                self.logger.info(
                    f"\n Before Computing KLD \n  {ram_usage()} \n {vram_usage()}"
                )

                # TODO: verify
                mb_kl = self.get_kl_divergence(
                    paths=paths_mb,
                    contexts=contexts_mb,
                    shifted_contexts=shifted_contexts_mb,
                    attention_mask=attention_mask,
                    action_log_probs=action_log_probs,
                    action_mask=action_mask_mb,
                )
                mb_kl *= self.config.kl_coeff

                self.tally.add_metric(path=["mb_kl_loss_terms"], metric=mb_kl.item())

                if self.config.log_kl_gradient_terms:
                    self.tally.add_metric(
                        path=["gradient_term_magnitudes", "kl"],
                        metric=self.get_gradient_magnitude(loss_term=mb_kl),
                    )

                loss += mb_kl

                self.logger.info(
                    f"\n After Computing KLD \n  {ram_usage()} \n {vram_usage()}"
                )

            # Normalize over number tokens generated
            # loss /= total_nb_action_tokens

            # Accumulate gradient
            self.accelerator.backward(loss)

            loss = 0.0

            # ensure gpu memory is freed
            del log_probs
            del contexts_mb
            del shifted_contexts_mb
            del credits_mb
            del action_mask_mb
            del attention_mask
            del logits
            del action_log_probs
            del rewarded_action_log_probs
            torch.cuda.empty_cache()

        # self.accelerator.sync_gradients

        # Clip gradients and take step
        if self.config.gradient_clipping is not None:
            grad_norm = self.accelerator.clip_grad_norm_(
                self.model.parameters(), self.config.gradient_clipping
            )
            # TODO: log at right place
            self.tally.add_metric(path=["gradient_norm"], metric=grad_norm.item())

        # TODO: log grad norm even if no grad clip

        # Take step
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Clear
        # TODO: verify
        self.accelerator.clear(self.model, self.optimizer)
        import gc

        gc.collect()
        torch.cuda.empty_cache()

        self.logger.info(f"\n After Reinforce Step \n  {ram_usage()} \n {vram_usage()}")

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

    @staticmethod
    def get_discounted_returns(
        rewards: np.ndarray, discount_factor: float
    ) -> np.ndarray:
        """
        Computes Monte Carlo discounted returns for a sequence of rewards.

        Args:
            rewards (np.ndarray): Array of rewards for each timestep.

        Returns:
            np.ndarray: Array of discounted returns.
        """
        T = rewards.shape[0]
        discounted_returns = np.zeros_like(rewards)
        accumulator = 0.0
        for t in reversed(range(T)):
            accumulator = rewards[t] + discount_factor * accumulator
            discounted_returns[t] = accumulator
        return discounted_returns

    @staticmethod
    def get_gen_adv_estimates(
        rewards: np.ndarray,
        value_estimates: np.ndarray,
        discount_factor: float,
        lambda_coef: float,
    ) -> np.ndarray:
        """
        Computes Generalized Advantage Estimates (GAE) for a sequence of rewards and value estimates.
        See https://arxiv.org/pdf/1506.02438 for details.

        Args:
            rewards (np.ndarray): Array of rewards (length T).
            value_estimates (np.ndarray): Array of value estimates (length T+1).

        Returns:
            np.ndarray: Array of GAE values.
        """
        T = rewards.size
        tds = rewards + lambda_coef * value_estimates[1:] - value_estimates[:-1]
        gaes = np.zeros_like(tds)
        acc = 0.0
        for t in reversed(range(T)):
            acc = tds[t] + lambda_coef * discount_factor * acc
            gaes[t] = acc
        return gaes

    def rewards_to_step_credits(
        self,
        batch_rewards: list[np.ndarray],
        batch_contexts: list[torch.Tensor],
        batch_state_end_flags: list[np.ndarray],
    ) -> list[np.ndarray]:
        """
        Converts per-step rewards into per-step credits (returns or advantages) for each batch element.
        Uses GAE if enabled, otherwise uses Monte Carlo returns.
        Optionally trains the critic if GAE is used.

        Args:
            batch_rewards (list[np.ndarray]): List of reward arrays for each batch element.
            batch_contexts (list[torch.Tensor]): List of context tensors for each batch element.
            batch_state_end_flags (list[np.ndarray]): List of state end flag arrays for each batch element.

        Returns:
            list[np.ndarray]: List of per-step credit arrays for each batch element.
        """

        B = len(batch_rewards)

        # Get MC discounted returns
        discounted_returns = []
        for i in range(B):
            rewards = batch_rewards[i]
            discounted_returns.append(
                self.get_discounted_returns(
                    rewards=rewards, discount_factor=self.config.discount_factor
                )
            )
            self.tally.add_metric(path=["discounted_returns"], metric=rewards)

        batch_credits = []
        # Use GAE

        if self.config.use_gae:
            # use & train critic
            critic_loss = 0.0
            for i in range(B):
                target = torch.Tensor(discounted_returns[i]).to(self.config.device)
                ctx = batch_contexts[i].to(self.config.device)
                end_flags = batch_state_end_flags[i]

                # critic causal attention up to end flags
                vals_estimate = self.critic(ctx.unsqueeze(0)).squeeze()[
                    end_flags == True
                ]
                detached_vals_estimate = (
                    vals_estimate.detach().to(torch.float32).to("cpu").numpy()
                )
                # If no infinite trajectory bootstrap, append V(Terminal State) = 0
                if detached_vals_estimate.size == batch_rewards[i].size:
                    detached_vals_estimate = np.append(detached_vals_estimate, 0)

                self.tally.add_metric(
                    path=["value_estimates"], metric=detached_vals_estimate
                )

                # Get GAE target
                target = (
                    self.get_gen_adv_estimates(
                        rewards=batch_rewards[i],
                        value_estimates=detached_vals_estimate,
                        discount_factor=self.config.discount_factor,
                        lambda_coef=self.config.gae_lambda_for_targets,
                    )
                    + detached_vals_estimate[:-1]
                )

                critic_loss += F.huber_loss(
                    input=vals_estimate[: target.size],
                    target=torch.Tensor(target).to(
                        device=vals_estimate.device, dtype=vals_estimate.dtype
                    ),
                )
                self.accelerator.backward(critic_loss)

                self.tally.add_metric(path=["critic_loss"], metric=critic_loss.item())
                critic_loss = 0.0

                # Get GAE Credit
                credits = self.get_gen_adv_estimates(
                    rewards=batch_rewards[i],
                    value_estimates=detached_vals_estimate,
                    discount_factor=self.config.discount_factor,
                    lambda_coef=self.config.gae_lambda_for_credits,
                )
                batch_credits.append(credits)

            self.critic_optimizer.step()
            self.critic_optimizer.zero_grad()

        # Simply use Monte Carlo
        else:
            batch_credits = discounted_returns

        return batch_credits

    def set_training_data(self, paths: list[str]) -> None:
        """
        Loads and processes training data from a list of JSON file paths.
        Extracts game IDs, contexts, rewards, action masks, and other metadata, and prepares step credits for training.

        Args:
            paths (list[str]): List of file paths to JSON conversation files.
        """

        all_agent_ids = []
        all_game_ids = []
        all_contexts = []
        all_rewards = []
        all_action_masks = []
        all_action_timestamps = []
        all_state_end_flags = []

        for filepath in paths:

            with open(filepath) as f:
                train_file = json.load(f)

            agent_id = train_file["agent_id"]
            game_id = train_file["game_id"]
            chat = train_file["chat"]

            all_agent_ids.append(agent_id)
            all_game_ids.append(game_id)

            (token_ids, rewards, action_mask, credit_mask, state_end_flags) = (
                process_training_chat(
                    tokenizer=self.tokenizer,
                    chat_history=chat,
                    end_at_last_state_flag=self.config.end_at_last_state_flag,
                )
            )

            all_contexts.append(token_ids)
            all_rewards.append(rewards)
            all_action_masks.append(action_mask)
            all_action_timestamps.append(credit_mask)
            all_state_end_flags.append(state_end_flags)

        self.paths = paths
        self.agent_ids = all_agent_ids
        self.game_ids = all_game_ids
        self.step_rewards = all_rewards
        self.contexts = all_contexts
        self.action_masks = all_action_masks
        self.action_timestamps = all_action_timestamps
        self.state_end_flags = all_state_end_flags

        self.step_credits = self.rewards_to_step_credits(
            batch_rewards=self.step_rewards,
            batch_contexts=self.contexts,
            batch_state_end_flags=self.state_end_flags,
        )

    def send_trainer_info(self) -> dict:
        """
        Prepares and returns reward shaping information (game IDs, step rewards, step credits) to be shared with opponent trainers.

        Returns:
            dict: Dictionary containing game IDs, step rewards, and step credits.
        """

        info = {
            "agent_ids": deepcopy(self.agent_ids),
            "game_ids": deepcopy(self.game_ids),
            "step_rewards": deepcopy(self.step_rewards),
            "step_credits": deepcopy(self.step_credits),
        }

        return info

    def use_co_trainer_info(self, co_trainer_info: dict) -> None:
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

    def set_token_credits(self) -> None:
        """
        Converts per-step credits into per-token credits for each batch element, based on action timestamps.
        Stores the result in self.token_credits.
        """
        B = len(self.step_credits)
        all_token_credits = []
        for i in range(B):
            token_credits = np.zeros(self.action_timestamps[i].shape)
            assert (
                np.max(self.action_timestamps[i]) == self.step_credits[i].size - 1
            ), "Number of steps does not match number of actions."
            for j, c in enumerate(self.step_credits[i]):
                token_credits[self.action_timestamps[i] == j] = c
            all_token_credits.append(torch.Tensor(token_credits))
        self.token_credits = all_token_credits

    def train(self) -> None:
        """
        Runs a single training iteration: sets token credits and applies a REINFORCE step using the current batch.
        """

        self.set_token_credits()

        self.apply_reinforce_step(
            paths=self.paths,
            contexts=self.contexts,
            credits=self.token_credits,
            action_masks=self.action_masks,
        )

    def export_training_metrics(self) -> None:
        """
        Saves and resets the collected training metrics using the tally object.
        """

        self.tally.save(path=self.config.logging_path)
        self.tally.reset()

    def export_optimizer_states(self) -> None:
        """
        Saves the optimizer states for both the main model and critic (if it exists).
        """
        try:
            os.makedirs(self.save_path, exist_ok=True)

            torch.save(self.optimizer.state_dict(), self.policy_optimizer_path)
            self.logger.info(
                f"Saved main optimizer state to {self.policy_optimizer_path}"
            )

            if self.critic_optimizer is not None:
                torch.save(
                    self.critic_optimizer.state_dict(), self.critic_optimizer_path
                )
                self.logger.info(
                    f"Saved critic optimizer state to {self.critic_optimizer_path}"
                )
        except Exception as e:
            self.logger.error(f"Error saving optimizer states: {str(e)}")
            raise
