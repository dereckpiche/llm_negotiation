"""
TODO: Add coefficients for losses (depend on total number of tokens or batch)
TODO: adapt reinforce step for torch.compile
TODO: add lr schedulers support
"""
import logging
import os
import sys
from abc import ABC, abstractmethod
from typing import Literal, Union

import numpy as np
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
    get_generalized_advantage_estimates,
    get_rloo_credits,
)
from mllm.training.tally_basic import Tally
from mllm.training.tally_tokenwise import ContextualizedTokenwiseTally
from mllm.training.tokenize_chats import *
from mllm.training.tokenize_chats import process_training_chat
from mllm.training.training_data_utils import *
from mllm.training.training_data_utils import (
    TrainingBatch,
    TrajectoryBatch,
    get_tokenwise_credits,
)
from mllm.utils.resource_context import resource_logger_context

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))


class BaseTrainer(ABC):
    """
    Trainer
    """

    def __init__(
        self,
        policy: AutoModelForCausalLM,
        policy_optimizer: torch.optim.Optimizer,
        critic: Union[AutoModelForCausalLM, None],
        critic_optimizer: Union[torch.optim.Optimizer, None],
        tokenizer: AutoTokenizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        critic_lr_scheduler: Union[torch.optim.lr_scheduler.LRScheduler, None],
        ######################################################################
        entropy_coeff: float,
        kl_coeff: float,
        gradient_clipping: Union[float, None],
        restrict_tokens: Union[list[str], None],
        mini_batch_size: int,
        use_gradient_checkpointing: bool,
        temperature: float,
        device: str,
        use_gae: bool,
        pg_loss_normalization: Literal["batch", "nb_tokens"],
        use_rloo: bool,
        skip_discounted_state_visitation: bool,
        gae_lambda_for_credits: float,
        gae_lambda_for_targets: float,
        discount_factor: float,
        enable_tokenwise_logging: bool,
        save_path: str,
        reward_normalizing_constant: float = 1.0,
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

        self.tokenizer = tokenizer
        # self.tokenizer.padding_side = "left"  # needed for flash attention
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.lr_scheduler = lr_scheduler
        self.accelerator = Accelerator()
        (
            self.policy,
            self.policy_optimizer,
            self.critic,
            self.critic_optimizer,
        ) = self.accelerator.prepare(policy, policy_optimizer, critic, critic_optimizer)

        self.critic_lr_scheduler = critic_lr_scheduler
        self.tally = Tally()

        if use_gradient_checkpointing == True:
            self.policy.gradient_checkpointing_enable(dict(use_reentrant=False))
            if critic is not None:
                self.critic.gradient_checkpointing_enable(dict(use_reentrant=False))

        self.save_path = save_path

        # Load states if already exist
        self.policy_optimizer_path = os.path.join(
            self.save_path, "policy_optimizer_state.pt"
        )
        if os.path.exists(self.policy_optimizer_path):
            logger.info(
                f"Loading policy optimizer state from {self.policy_optimizer_path}"
            )
            self.policy_optimizer.load_state_dict(
                torch.load(self.policy_optimizer_path)
            )

        self.critic_optimizer_path = os.path.join(
            self.save_path, "critic_optimizer_state.pt"
        )
        if (
            os.path.exists(self.critic_optimizer_path)
            and self.critic_optimizer is not None
        ):
            logger.info(
                f"Loading critic optimizer state from {self.critic_optimizer_path}"
            )
            self.critic_optimizer.load_state_dict(
                torch.load(self.critic_optimizer_path)
            )
        self.device = self.accelerator.device
        self.entropy_coeff = entropy_coeff
        self.kl_coeff = kl_coeff
        self.gradient_clipping = gradient_clipping
        self.restrict_tokens = restrict_tokens
        self.mini_batch_size = mini_batch_size
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.temperature = temperature
        self.use_gae = use_gae
        self.use_rloo = use_rloo
        self.skip_discounted_state_visitation = skip_discounted_state_visitation
        self.gae_lambda_for_credits = gae_lambda_for_credits
        self.gae_lambda_for_targets = gae_lambda_for_targets
        self.discount_factor = discount_factor
        self.enable_tokenwise_logging = enable_tokenwise_logging
        self.reward_normalizing_constant = reward_normalizing_constant
        self.pg_loss_normalization = pg_loss_normalization
        # Common containers used by all trainers
        self.training_data: dict = {}
        self.debug_path_list: list[str] = []
        self.policy_gradient_data = None
        self.tokenwise_tally = None

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

        if self.restrict_tokens is not None:
            allowed_token_ids = []
            for token in self.restrict_tokens:
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

    # def get_gradient_magnitude(self, loss_term: torch.Tensor) -> float:
    #     """
    #     Computes the L2 norm of the gradients of the given loss term with respect to the model parameters.

    #     Args:
    #         loss_term (torch.Tensor): The loss tensor to compute gradients for.

    #     Returns:
    #         float: The L2 norm of the gradients, or 0.0 if no gradients are present.
    #     """
    #     with torch.no_grad():
    #         grads = torch.autograd.grad(
    #             loss_term,
    #             [p for p in self.policy.parameters() if p.requires_grad],
    #             retain_graph=True,
    #             allow_unused=True,
    #         )
    #         grads = [g for g in grads if g is not None]
    #         if not grads:
    #             return torch.tensor(0.0, device=loss_term.device)
    #         return torch.norm(torch.stack([g.norm(2) for g in grads])).item()

    def apply_reinforce_step(
        self,
        training_batch: TrainingBatch,
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
        with resource_logger_context(logger, "Apply reinforce step"):
            self.policy.train()
            mb_size = self.mini_batch_size
            nb_rollouts = len(training_batch)
            self.tally.add_metric(path=["nb_rollouts"], metric=nb_rollouts)

            # Get total number of tokens generated
            total_tokens_generated = 0
            for att_mask in training_batch.batch_action_mask:
                total_tokens_generated += att_mask.sum()

            # Obtain loss normalization
            if self.pg_loss_normalization == "nb_tokens":
                normalization_factor = total_tokens_generated
            elif self.pg_loss_normalization == "batch":
                normalization_factor = np.ceil(nb_rollouts / mb_size).astype(int)
            else:
                raise ValueError(
                    f"Invalid pg_loss_normalization: {self.pg_loss_normalization}"
                )

            # Gradient accumulation for each mini-batch
            for mb in range(0, nb_rollouts, mb_size):
                loss = 0.0
                training_mb = training_batch[mb : mb + mb_size]
                training_mb = training_mb.get_padded_tensors()
                training_mb.to(self.device)
                tokens_mb, action_mask_mb, credits_mb = (
                    training_mb.batch_input_ids,
                    training_mb.batch_action_mask,
                    training_mb.batch_credits,
                )

                # Next token prediction
                contexts_mb = tokens_mb[:, :-1]
                shifted_contexts_mb = tokens_mb[:, 1:]
                action_mask_mb = action_mask_mb[:, 1:]
                credits_mb = credits_mb[:, 1:]

                if self.enable_tokenwise_logging:
                    self.tokenwise_tally.set_action_mask(action_mask=action_mask_mb)
                    self.tokenwise_tally.set_range(range=(mb, mb + mb_size))
                    self.tokenwise_tally.add_contexts(contexts=contexts_mb)
                    self.tokenwise_tally.add_data(
                        metric_id="next_token",
                        metrics=shifted_contexts_mb,
                        to_tids=True,
                    )

                if self.enable_tokenwise_logging:
                    self.tokenwise_tally.add_data(
                        metric_id="next_token_credit", metrics=credits_mb
                    )

                # Forward pass + cast to FP-32 for higher prec.
                # TODO: create attention mask if not relying on default (assume causal llm)
                logits = self.policy(input_ids=contexts_mb)[0]  # (B, S, V)

                # Mask non-restricted tokens
                if self.restrict_tokens is not None:
                    logits = self.mask_non_restricted_token_logits(logits)

                logits /= self.temperature  # (B, S, V)

                # Compute new log probabilities
                log_probs = F.log_softmax(logits, dim=-1)  # (B, S, V)

                # Get log probabilities of actions taken during rollouts
                action_log_probs = log_probs.gather(
                    dim=-1, index=shifted_contexts_mb.unsqueeze(-1)
                ).squeeze(
                    -1
                )  # (B, S)

                if self.enable_tokenwise_logging:
                    self.tokenwise_tally.add_data(
                        metric_id="next_token_log_prob",
                        metrics=action_log_probs,
                    )
                    self.tokenwise_tally.add_data(
                        metric_id="next_token_prob",
                        metrics=torch.exp(action_log_probs),
                    )
                    top_k_indices = torch.topk(logits, k=5, dim=-1).indices
                    self.tokenwise_tally.add_data(
                        metric_id=f"top_{5}_tids",
                        metrics=top_k_indices,
                        to_tids=True,
                    )
                    self.tokenwise_tally.add_data(
                        metric_id=f"top_{5}_probs",
                        metrics=torch.exp(log_probs).gather(
                            dim=-1, index=top_k_indices
                        ),
                    )

                rewarded_action_log_probs = (
                    action_mask_mb * credits_mb * action_log_probs
                )
                # (B, S)

                if self.enable_tokenwise_logging:
                    self.tokenwise_tally.add_data(
                        metric_id="next_token_clogπ",
                        metrics=rewarded_action_log_probs,
                    )

                # Add value term to loss
                if self.pg_loss_normalization == "batch":
                    nb_act_tokens = action_mask_mb.sum()
                    mb_value = -rewarded_action_log_probs.sum() / nb_act_tokens
                else:
                    mb_value = -rewarded_action_log_probs.sum()

                # if self.enable_tokenwise_logging:
                #     self.tally.add_metric(
                #         path=["gradient_term_magnitudes", "value"],
                #         metric=self.get_gradient_magnitude(loss_term=mb_value),
                #     )
                loss += mb_value
                self.tally.add_metric(
                    path=["loss_mb_total", "value_mb_total"], metric=mb_value.item()
                )
                # -------------------------------------------------
                # Entropy Regularization
                # -------------------------------------------------
                if self.entropy_coeff != 0.0:
                    token_entropy_terms = -F.softmax(logits, dim=-1) * F.log_softmax(
                        logits, dim=-1
                    )
                    # (B, S, T)
                    # We only take the entropy of actions
                    token_entropy_terms *= action_mask_mb[:, :, None]
                    mb_entropy = token_entropy_terms.sum(dim=-1)
                    if self.enable_tokenwise_logging:
                        self.tally.add_contextualized_token_metrics(
                            metric_id="entropy",
                            metrics=mb_entropy,
                        )

                    if self.pg_loss_normalization == "batch":
                        nb_act_tokens = action_mask_mb.sum()
                        mb_entropy = -mb_entropy.sum() / nb_act_tokens
                    else:
                        mb_entropy = -mb_entropy.sum()

                    mb_entropy *= self.entropy_coeff
                    self.tally.add_metric(
                        path=["loss_mb_total", "entropy_mb_total"],
                        metric=mb_entropy.item(),
                    )

                    # if self.enable_tokenwise_logging:
                    #     self.tally.add_metric(
                    #         path=["gradient_term_magnitudes", "entropy"],
                    #         metric=self.get_gradient_magnitude(loss_term=mb_entropy),
                    #     )
                    loss += mb_entropy

                # -------------------------------------------------
                # KL-DIVERGENCE
                # -------------------------------------------------
                if self.kl_coeff != 0.0:
                    with torch.no_grad():
                        with self.policy.disable_adapter():
                            ref_model_logits = self.policy(
                                input_ids=contexts_mb,  # attention_mask=attention_mask
                            )[0]
                    ref_model_logits = ref_model_logits / self.temperature
                    # (B, S, V)
                    ref_model_logits = self.mask_non_restricted_token_logits(
                        logits=ref_model_logits
                    )
                    # (B, S, V)
                    ref_model_log_probs = F.log_softmax(ref_model_logits, dim=-1)
                    # (B, S, V)
                    ref_model_action_log_probs = ref_model_log_probs.gather(
                        dim=-1, index=shifted_contexts_mb.unsqueeze(-1)
                    ).squeeze(
                        -1
                    )  # (B,S)
                    # Approximating KL Divergence (see refs in docstring)
                    # Ref 1: http://joschu.net/blog/kl-approx.html
                    # Ref 2: https://github.dev/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py#L1332
                    kl_div = (
                        torch.exp(ref_model_action_log_probs - action_log_probs)
                        - (ref_model_action_log_probs - action_log_probs)
                        - 1
                    )

                    if self.enable_tokenwise_logging:
                        self.tally.add_contextualized_token_metrics(
                            metric_id="kl",
                            metrics=kl_div,
                        )

                    # We only care about KLD of action tokens
                    kl_div *= action_mask_mb
                    mb_kl = kl_div.sum()

                    mb_kl *= self.kl_coeff

                    self.tally.add_metric(
                        path=["mb_kl_loss_terms"], metric=mb_kl.item()
                    )

                    if self.enable_tokenwise_logging:
                        self.tally.add_metric(
                            path=["gradient_term_magnitudes", "kl"],
                            metric=self.get_gradient_magnitude(loss_term=mb_kl),
                        )

                    loss += mb_kl

                # Accumulate gradient
                loss /= normalization_factor
                self.accelerator.backward(loss)

                # ensure gpu memory is freed
                del training_mb
                del log_probs
                del logits
                del loss
                del action_log_probs
                del rewarded_action_log_probs

            logger.info(
                f"Accumulated the policy gradient loss for {total_tokens_generated} tokens."
            )

            # Clip gradients and take step
            if self.gradient_clipping is not None:
                grad_norm = self.accelerator.clip_grad_norm_(
                    self.policy.parameters(), self.gradient_clipping
                )
                # TODO: log at right place
                self.tally.add_metric(path=["gradient_norm"], metric=grad_norm.item())

            # TODO: log grad norm even if no grad clip

            # Take step
            self.policy_optimizer.step()
            self.policy_optimizer.zero_grad()

            # Clear
            # TODO: verify
            self.accelerator.clear(self.policy, self.policy_optimizer)
            import gc

            gc.collect()
            torch.cuda.empty_cache()

    def get_advantages_with_critic_gradient_accumulation(
        self, trajectories: TrajectoryBatch, loss_scaling_factor: float = 1
    ) -> torch.FloatTensor:
        """
        TOWRITE
        Uses GAE if enabled, otherwise uses Monte Carlo returns.
        Optionally trains the critic if GAE is used.
        Returns:
            credits: NestedFloatTensors
        """

        mb_size = self.mini_batch_size
        batch_size = trajectories.rollout_ids.shape[0]
        # self.tally.add_metric(path=["discounted_returns"], metric=rewards)

        if self.use_gae:
            credits = []

            # For each minibatch
            for mb in range(0, batch_size, mb_size):
                trajectory_mb = trajectories[mb : mb + mb_size]
                trajectory_mb.to(self.device)
                rewards_mb = trajectory_mb.batch_rewards
                self.tally.add_metric(path=["mb_rewards"], metric=rewards_mb)
                # use & train critic
                (
                    tokens_mb,
                    state_ends_mask_mb,
                    timestep_counts,
                ) = trajectory_mb.get_padded_tensors_for_critic()

                # critic causal attention up to end flags
                vals_estimate_full = self.critic(tokens_mb)
                if vals_estimate_full.dim() == 3:
                    vals_estimate_full = vals_estimate_full.squeeze(-1)
                # Select only positions where states end, per sample → list of (jT,)
                B = tokens_mb.shape[0]
                vals_list = [
                    vals_estimate_full[b][state_ends_mask_mb[b]] for b in range(B)
                ]
                # Pad to (B, max_jT)
                vals_estimate_mb = pad_sequence(
                    vals_list, batch_first=True, padding_value=0.0
                )
                dtype = vals_estimate_mb.dtype
                # Ensure rewards have the same dtype as value estimates; pad to same (B, max_jT)
                rewards_mb = pad_sequence(
                    rewards_mb, batch_first=True, padding_value=0.0
                ).to(dtype=dtype)

                det_vals_estimate_mb = vals_estimate_mb.detach()
                self.tally.add_metric(
                    path=["mb_value_estimates_critic"], metric=det_vals_estimate_mb
                )

                # If no infinite trajectory bootstrap, append V(Terminal State) = 0
                # TODO: use alternative approach!
                if det_vals_estimate_mb.shape[1] == rewards_mb.shape[1]:
                    Bsize = det_vals_estimate_mb.shape[0]
                    device = det_vals_estimate_mb.device
                    dtype = det_vals_estimate_mb.dtype
                    det_vals_estimate_mb = torch.cat(
                        [
                            det_vals_estimate_mb,
                            torch.zeros((Bsize, 1), device=device, dtype=dtype),
                        ],
                        dim=1,
                    )

                # self.tally.add_metric(
                #     path=["value_estimates"], metric=detached_vals_estimate
                # )

                # Get GAE target
                gae_targets = get_generalized_advantage_estimates(
                    rewards=rewards_mb,
                    value_estimates=det_vals_estimate_mb,
                    discount_factor=self.discount_factor,
                    lambda_coef=self.gae_lambda_for_targets,
                )
                # Ensure dtype consistency
                gae_targets = gae_targets.to(dtype=dtype)
                targets = gae_targets + det_vals_estimate_mb[:, :-1]
                self.tally.add_metric(path=["mb_targets_critic"], metric=targets)

                loss = loss_scaling_factor * F.huber_loss(
                    input=vals_estimate_mb,
                    target=targets,
                )
                self.tally.add_metric(path=["mb_critic_loss"], metric=loss.item())
                self.accelerator.backward(loss)

                # self.tally.add_metric(path=["critic_loss"], metric=loss.item())

                # Get GAE Credit
                credits_padded = get_generalized_advantage_estimates(
                    rewards=rewards_mb,
                    value_estimates=det_vals_estimate_mb,
                    discount_factor=self.discount_factor,
                    lambda_coef=self.gae_lambda_for_credits,
                )
                self.tally.add_metric(path=["mb_gae_credits"], metric=credits_padded)
                # Ensure dtype consistency for credits
                credits_padded = credits_padded.to(dtype=dtype)

                # Get jagged back using timestep_counts from get_padded_tensors_for_critic()
                credits.extend(
                    [credits_padded[i, : timestep_counts[i]] for i in range(B)]
                )

            # return list-of-tensors

        else:
            batch_rewards = trajectories.batch_rewards
            self.tally.add_metric(path=["batch_rewards"], metric=batch_rewards)
            lengths = [len(c) for c in batch_rewards]
            padded_rewards = pad_sequence(
                batch_rewards, batch_first=True, padding_value=0.0
            )
            padded_credits = get_discounted_returns(
                rewards=padded_rewards,
                discount_factor=self.discount_factor,
                tally=self.tally,
            )
            if self.use_rloo:
                is_grouped_by_rng = (
                    trajectories.crn_ids.unique().shape[0]
                    != trajectories.crn_ids.shape[0]
                )
                if is_grouped_by_rng:
                    for crn_id in trajectories.crn_ids.unique():
                        rng_mask = trajectories.crn_ids == crn_id
                        rng_credits = padded_credits[rng_mask]
                        rng_credits, _ = get_rloo_credits(
                            credits=rng_credits, tally=self.tally
                        )
                        padded_credits[rng_mask] = rng_credits
                else:
                    padded_credits, _ = get_rloo_credits(
                        credits=padded_credits, tally=self.tally
                    )
            credits = [
                padded_credits[i, : lengths[i]] for i in range(padded_credits.shape[0])
            ]
        return credits

    @abstractmethod
    def set_agent_trajectory_data(
        self, agent_id: str, roots: list[RolloutTreeRootNode]
    ) -> None:
        """
        TOWRITE
        """
        pass

    def set_trajectory_data(
        self, rollout_trees: list[RolloutTreeRootNode], agent_ids: list[str]
    ) -> None:
        """
        TOWRITE
        """
        for agent_id in agent_ids:
            self.set_agent_trajectory_data(agent_id, rollout_trees)

    @abstractmethod
    def share_advantage_data(self) -> list[AdvantagePacket]:
        pass

    @abstractmethod
    def receive_advantage_data(self, advantage_packets: list[AdvantagePacket]) -> None:
        pass

    def set_policy_gradient_data(self) -> None:
        """
        Already set earlier # TODO: make it separate and clean
        """
        self.policy_gradient_data = None
        # Track row id order aligned with concatenation
        concat_crn_ids = []
        concat_rollout_ids = []
        concat_agent_ids = []
        for agent_id, trajectory_batch in self.training_data.items():
            tokenwise_batch_credits = get_tokenwise_credits(
                batch_timesteps=trajectory_batch.batch_timesteps,
                batch_credits=trajectory_batch.batch_credits,
            )
            policy_gradient_data = TrainingBatch(
                rollout_ids=trajectory_batch.rollout_ids,
                batch_input_ids=trajectory_batch.batch_input_ids,
                batch_action_mask=trajectory_batch.batch_action_mask,
                batch_credits=tokenwise_batch_credits,
            )
            if self.policy_gradient_data is None:
                self.policy_gradient_data = policy_gradient_data
            else:
                self.policy_gradient_data.append(policy_gradient_data)

            concat_crn_ids.append(trajectory_batch.crn_ids)
            concat_rollout_ids.append(trajectory_batch.rollout_ids)
            concat_agent_ids.extend(trajectory_batch.agent_ids)

        self.tokenwise_tally = ContextualizedTokenwiseTally(
            tokenizer=self.tokenizer,
            paths=self.debug_path_list,
        )

        # Register row ids once in the same order used to build policy_gradient_data
        if concat_rollout_ids:
            try:
                crn_all = (
                    torch.cat(concat_crn_ids)
                    if len(concat_crn_ids) > 1
                    else concat_crn_ids[0]
                )
                rid_all = (
                    torch.cat(concat_rollout_ids)
                    if len(concat_rollout_ids) > 1
                    else concat_rollout_ids[0]
                )
            except Exception:
                crn_all = concat_crn_ids[0]
                rid_all = concat_rollout_ids[0]
            self.tally.add_row_ids(
                crn_ids=crn_all, rollout_ids=rid_all, agent_ids=concat_agent_ids
            )

    def train(self) -> None:
        """
        TOWRITE
        """
        assert self.policy_gradient_data is not None, "Policy gradient data is not set"
        if self.critic_optimizer is not None:
            self.critic.train()
            self.critic_optimizer.zero_grad()
        self.apply_reinforce_step(training_batch=self.policy_gradient_data)

    def export_training_tally(self, identifier: str, folder: str) -> None:
        """
        Saves and resets the collected training metrics using the tally object.
        """
        os.makedirs(folder, exist_ok=True)
        self.tally.save(identifier=identifier, folder=folder)
        self.tokenwise_tally.save(
            path=os.path.join(folder, f"{identifier}_tokenwise.csv")
        )
        self.tally.reset()
        self.tokenwise_tally = None
        self.debug_path_list = []

    def export_optimizer_states(self) -> None:
        """
        Saves the optimizer states for both the main model and critic (if it exists).
        """
        try:
            os.makedirs(self.save_path, exist_ok=True)

            torch.save(self.policy_optimizer.state_dict(), self.policy_optimizer_path)
            logger.info(f"Saved main optimizer state to {self.policy_optimizer_path}")

            if self.critic_optimizer is not None:
                torch.save(
                    self.critic_optimizer.state_dict(), self.critic_optimizer_path
                )
                logger.info(
                    f"Saved critic optimizer state to {self.critic_optimizer_path}"
                )
        except Exception as e:
            logger.error(f"Error saving optimizer states: {str(e)}")
            raise
