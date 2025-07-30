"""
TODO: Add coefficients for losses (depend on total number of tokens or batch)
"""


import logging
import os
import random
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from pandas._libs.tslibs.offsets import CBMonthBegin
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer

from mllm.training.tally import RtTally
from mllm.training.training_data_utils import TrajectoryBatch, TrainingBatch
from mllm.training.tokenize_chats import process_training_chat
from mllm.utils.common_imports import *
from mllm.utils.time_and_memory_utils import *
from mllm.utils.print_logger import *
from typing import Union, Tuple, Any

from typing import Union
from peft import LoraConfig
from dataclasses import dataclass
from mllm.markov_games.rollout_tree import RolloutTreeRootNode
from mllm.training.credit_methods import get_discounted_returns, get_generalized_advantage_estimates
from mllm.training.training_data_utils import *
from mllm.training.tokenize_chats import *
from mllm.markov_games.rollout_tree import *

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))

@dataclass
class BaseTrainerConfig:
    entropy_coeff: float # Coefficient of the entropy term in the loss.
    kl_coeff: float # Coefficient of the KL-divergence term in the loss.
    gradient_clipping: Union[float, None] # Maximum norm of the gradient component before it gets clipped.
    restrict_tokens: Union[list[str], None]
    mini_batch_size: int # The number of conversations/trajectories we backpropagate through at once. This only affects the GPU usage.
    use_gradient_checkpointing: bool
    temperature: float
    device: str

    # Regular credit assignment
    use_gae: bool
    gae_lambda_for_credits: float
    gae_lambda_for_targets: float
    discount_factor: float
    end_at_last_state_flag: bool # False

    # Opponent Shaping
    use_sum_credits: bool
    use_advantage_alignment: bool
    ad_align_normalize_advantages: bool
    ad_align_force_coop_first_step: bool
    use_sign_in_ad_align: bool
    ad_align_clipping: float
    use_time_regularization_in_ad_align: bool
    use_variance_regularization_in_ad_align: bool
    ad_align_beta: float

    # Regular logging
    log_entropy_gradient_terms: bool = False
    log_kl_gradient_terms: bool = False
    log_value_gradient_terms: bool = False

    # Contextualized logging
    debug_mode: bool = False


class BaseTrainer:
    """
    Trainer
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
        # config: BaseTrainerConfig,
        save_path: str,
        **kwargs
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
        self.config = BaseTrainerConfig(**kwargs)

        # TODO: add lr schedulers
        model.train()
        self.tokenizer = tokenizer
        # self.tokenizer.padding_side = "left"  # needed for flash attention
        if self.tokenizer.pad_token_id is None: self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
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
        self.device = self.accelerator.device

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
        training_batch: TrainingBatch,
        loss_scaling_factor: float = 1
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
        mb_size = self.config.mini_batch_size
        nb_rollouts = len(training_batch)
        self.tally.add_metric(path=["nb_rollouts"], metric=nb_rollouts)

        # Count total number of tokens trained on
        # total_nb_action_tokens = 0
        # for am in action_masks:
        #     nb_tokens = am.shape[0]
        #     self.tally.add_metric(path=["nb_tokens", "action_tokens"], metric=nb_tokens)
        #     total_nb_action_tokens += nb_tokens

        # self.tally.add_metric(
        #     path=["nb_tokens", "batch_action_tokens"], metric=total_nb_action_tokens
        # )

        for mb in range(0, nb_rollouts, mb_size):
            loss = 0.0
            training_mb = training_batch[mb: mb + mb_size]
            training_mb = training_mb.get_padded_tensors()
            training_mb.to(self.device)
            tokens_mb, action_mask_mb, credits_mb = training_mb.batch_input_ids, training_mb.batch_action_mask, training_mb.batch_credits

            # Next token prediction
            contexts_mb = tokens_mb[:, :-1]
            shifted_contexts_mb = tokens_mb[:, 1:]
            action_mask_mb = action_mask_mb[:, 1:]
            credits_mb = credits_mb[:, 1:]

            if self.config.debug_mode:
                self.tally.add_contextualized_token_metrics(
                    paths=paths_mb,
                    metric_id="next_token_credit",
                    contexts=shifted_contexts_mb,
                    metrics=credits_mb,
                    action_mask=action_mask_mb,
                )

            # Forward pass + cast to FP-32 for higher prec.
            # TODO: create attention mask if not relying on default (assume causal llm)
            logits = self.model(input_ids=contexts_mb)[0]  # (B, S, V)

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

            if self.config.debug_mode:
                self.tally.add_contextualized_token_metrics(
                    paths=paths_mb,
                    metric_id="next_token_log_prob",
                    contexts=shifted_contexts_mb,
                    metrics=action_log_probs,
                    action_mask=action_mask_mb,
                )
                self.tally.add_contextualized_token_metrics(
                    paths=paths_mb,
                    metric_id="next_token_prob",
                    contexts=shifted_contexts_mb,
                    metrics=torch.exp(action_log_probs),
                    action_mask=action_mask_mb,
                )
                top_k_indices = torch.topk(
                    logits, k=5, dim=-1
                ).indices
                self.tally.add_contextualized_token_metrics(
                    paths=paths_mb,
                    metric_id=f"top_{5}_tids",
                    contexts=shifted_contexts_mb,
                    metrics=top_k_indices,
                    action_mask=action_mask_mb,
                    to_tids=True,
                )
                self.tally.add_contextualized_token_metrics(
                    paths=paths_mb,
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

            if self.config.debug_mode:
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

            if self.config.debug_mode:
                self.tally.add_metric(
                    path=["gradient_term_magnitudes", "value"],
                    metric=self.get_gradient_magnitude(loss_term=mb_value),
                )
            loss += mb_value

            # -------------------------------------------------
            # Entropy Regularization
            # -------------------------------------------------
            if self.config.entropy_coeff != 0.0:

                self.logger.info(
                    f"\n Before Computing Entropy \n  {ram_usage()} \n {vram_usage()}"
                )

                token_entropy_terms = -F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)
                # (B, S, T)
                # We only take the entropy of actions
                token_entropy_terms *= action_mask_mb[:, :, None]
                # if self.config.debug_mode:
                #     self.tally.add_contextualized_token_metrics(
                #         game_ids=paths,
                #         metric_id="entropy",
                #         contexts=shifted_contexts,
                #         metrics=token_entropy_terms.sum(dim=-1),
                #         action_mask=action_mask,
                #     )

                mb_entropy = token_entropy_terms.sum()
                del token_entropy_terms
                mb_entropy *= self.config.entropy_coeff
                self.tally.add_metric(
                    path=["loss_mb_total", "entropy_mb_total"], metric=mb_entropy.item()
                )

                if self.config.debug_mode:
                    self.tally.add_metric(
                        path=["gradient_term_magnitudes", "entropy"],
                        metric=self.get_gradient_magnitude(loss_term=mb_entropy),
                    )
                loss += mb_entropy

                self.logger.info(
                    f"\n After Computing Entropy \n  {ram_usage()} \n {vram_usage()}"
                )

            # -------------------------------------------------
            # KL-DIVERGENCE
            # -------------------------------------------------
            if self.config.kl_coeff != 0.0:
                self.logger.info(
                    f"\n Before Computing KLD \n  {ram_usage()} \n {vram_usage()}"
                )
                with torch.no_grad():
                    with self.model.disable_adapter():
                        ref_model_logits = self.model(
                            input_ids=contexts_mb, # attention_mask=attention_mask
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

                # if self.config.debug_mode:
                #     self.tally.add_contextualized_token_metrics(
                #         game_ids=game_ids,
                #         metric_id="kl",
                #         contexts=shifted_contexts,
                #         metrics=kl_div,
                #         action_mask=action_mask,
                #     )

                # We only care about KLD of action tokens
                kl_div *= action_mask_mb
                mb_kl = kl_div.sum()


                mb_kl *= self.config.kl_coeff

                self.tally.add_metric(path=["mb_kl_loss_terms"], metric=mb_kl.item())

                if self.config.debug_mode:
                    self.tally.add_metric(
                        path=["gradient_term_magnitudes", "kl"],
                        metric=self.get_gradient_magnitude(loss_term=mb_kl),
                    )

                loss += mb_kl

                self.logger.info(
                    f"\n After Computing KLD \n  {ram_usage()} \n {vram_usage()}"
                )

            # Accumulate gradient
            loss *= loss_scaling_factor
            self.accelerator.backward(loss)


            # ensure gpu memory is freed
            # del log_probs
            # del contexts_mb
            # del shifted_contexts_mb
            # del credits_mb
            # del action_mask_mb
            # del attention_mask
            # del logits
            # del action_log_probs
            # del rewarded_action_log_probs
            # torch.cuda.empty_cache()

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


    def get_advantages_with_critic_gradient_accumulation(
        self,
        trajectories: TrajectoryBatch,
        loss_scaling_factor: float = 1
    ) -> torch.FloatTensor:
        """
        TOWRITE
        Uses GAE if enabled, otherwise uses Monte Carlo returns.
        Optionally trains the critic if GAE is used.
        Returns:
            credits: NestedFloatTensors
        """

        mb_size = self.config.mini_batch_size
        batch_size = trajectories.rollout_ids.shape[0]
        # self.tally.add_metric(path=["discounted_returns"], metric=rewards)

        if self.config.use_gae:

            credits = []

            # For each minibatch
            for mb in range(0, batch_size, mb_size):

                trajectory_mb = trajectories[mb:mb+mb_size]
                trajectory_mb.to(self.device)
                rewards_mb = trajectory_mb.batch_rewards

                # use & train critic
                tokens_mb, state_ends_mask_mb, _ = trajectory_mb.get_padded_tensors_for_critic()

                # critic causal attention up to end flags
                vals_estimate_mb = self.critic(tokens_mb)
                vals_estimate_mb = torch.nested.masked_select(vals_estimate_mb, state_ends_mask_mb) # Get value estimates only where states end
                if vals_estimate_mb.dim() == 1: vals_estimate_mb = vals_estimate_mb.unsqueeze(0)

                # Get padded tensors
                jagged_lengths = vals_estimate_mb.offsets().diff()
                padded_shape = (vals_estimate_mb.shape[0], torch.max(jagged_lengths))
                vals_estimate_mb = torch.nested.to_padded_tensor(vals_estimate_mb, padding=0.0, output_size=padded_shape)
                rewards_mb = torch.nested.to_padded_tensor(rewards_mb, padding=0.0, output_size=padded_shape)

                det_vals_estimate_mb = vals_estimate_mb.detach()

                # If no infinite trajectory bootstrap, append V(Terminal State) = 0
                # TODO: use alternative approach!
                if det_vals_estimate_mb.numel() == rewards_mb.numel():
                        B = det_vals_estimate_mb.shape[0]
                        device = det_vals_estimate_mb.device
                        det_vals_estimate_mb = torch.cat([det_vals_estimate_mb, torch.zeros((B, 1), device=device)], dim=1)

                # self.tally.add_metric(
                #     path=["value_estimates"], metric=detached_vals_estimate
                # )

                # Get GAE target
                targets = (
                    get_generalized_advantage_estimates(
                        rewards=rewards_mb,
                        value_estimates=det_vals_estimate_mb,
                        discount_factor=self.config.discount_factor,
                        lambda_coef=self.config.gae_lambda_for_targets,
                    )
                    + det_vals_estimate_mb[:, :-1]
                )

                loss = loss_scaling_factor * F.huber_loss(
                    input=vals_estimate_mb,
                    target=targets,
                )
                self.accelerator.backward(loss)

                # self.tally.add_metric(path=["critic_loss"], metric=loss.item())

                # Get GAE Credit
                credits_mb = get_generalized_advantage_estimates(
                    rewards=rewards_mb,
                    value_estimates=det_vals_estimate_mb,
                    discount_factor=self.config.discount_factor,
                    lambda_coef=self.config.gae_lambda_for_credits,
                )

                # Get jagged back
                # import pdb; pdb.set_trace()
                credits_mb = torch.nested.narrow(credits_mb, dim=1, start=0, length=jagged_lengths, layout=torch.jagged)
                credits.extend(credits_mb.unbind())

            credits = torch.nested.nested_tensor(credits, dtype=torch.float32, layout=torch.jagged)

        else:
            batch_rewards = trajectories.batch_rewards
            credits = [ get_discounted_returns(
                rewards=rewards,
                discount_factor=self.config.discount_factor
            )
            for rewards in batch_rewards ]
            credits = torch.nested.nested_tensor(credits, dtype=torch.float32, layout=torch.jagged)

        return credits


    def set_policy_gradient_data(self, rollout_trees: list[RolloutTreeRootNode], agent_ids: list[str]) -> None:
        """
        TOWRITE
        """

        # Tensorize Chats
        rollout_ids = []
        chats = []

        def get_chat_list_and_rewards(
            agent_id: str, root : RolloutTreeRootNode) -> Tuple[list[ChatTurn], torch.FloatTensor]:
            """
            TOWRITE
            """
            # TODO; extend for all trees, not just linear
            current_node = root.child
            chat = []
            rewards = []
            while current_node is not None:
                reward : float = current_node.step_log.simulation_step_log.rewards[agent_id]
                rewards.append(reward)
                chat_turns: list[ChatTurn] = current_node.step_log.action_logs[agent_id].chat_turns
                chat.extend(chat_turns)
                current_node = current_node.child
            return chat, torch.FloatTensor(rewards)


        batch_input_ids = []
        batch_action_mask = []
        batch_timesteps = []
        batch_state_ends_mask = []
        batch_rewards = []
        for agent_id in agent_ids:
            for root in rollout_trees:
                rollout_ids.append(root.id)
                chat, rewards = get_chat_list_and_rewards(agent_id=agent_id, root=root)
                (
                    input_ids,
                    action_mask,
                    timesteps,
                    state_ends_mask,
                ) = process_training_chat(
                    tokenizer=self.tokenizer,
                    chat_history=chat
                )
                batch_input_ids.append(input_ids)
                batch_action_mask.append(action_mask)
                batch_timesteps.append(timesteps)
                batch_state_ends_mask.append(state_ends_mask)
                batch_rewards.append(rewards)

        trajectory_batch = TrajectoryBatch(
            rollout_ids = torch.Tensor(rollout_ids),
            batch_input_ids = torch.nested.nested_tensor(batch_input_ids, layout=torch.jagged),
            batch_action_mask = torch.nested.nested_tensor(batch_action_mask, layout=torch.jagged),
            batch_timesteps = torch.nested.nested_tensor(batch_timesteps, layout=torch.jagged),
            batch_state_ends_mask = torch.nested.nested_tensor(batch_state_ends_mask, layout=torch.jagged),
            batch_rewards = torch.nested.nested_tensor(batch_rewards, layout=torch.jagged)
        )

        # Get Advantages & Train Critic
        batch_advantages: torch.FloatTensor = self.get_advantages_with_critic_gradient_accumulation(trajectory_batch)
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad()


        batch_advantages = get_tokenwise_credits(
            batch_timesteps = trajectory_batch.batch_timesteps,
            batch_credits = batch_advantages
        )

        # Get training batch and apply single step
        self.training_batch = TrainingBatch(
            rollout_ids = trajectory_batch.rollout_ids,
            batch_input_ids = trajectory_batch.batch_input_ids,
            batch_action_mask = trajectory_batch.batch_action_mask,
            batch_credits = batch_advantages
        )


    def train(self) -> None:
        """
        TOWRITE
        """
        self.apply_reinforce_step(training_batch=self.training_batch)


    def set_token_credits(self) -> None:
        """
        Converts per-step credits into per-token credits for each batch element, based on action timestamps.
        Stores the result in self.token_credits.
        """


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
