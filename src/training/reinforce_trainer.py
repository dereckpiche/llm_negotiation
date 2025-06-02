import logging
import os
import random
from collections import defaultdict
from contextlib import contextmanager, nullcontext

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from training.reinforce_trainer_config import RtConfig
from training.reinforce_trainer_tally import RtTally
from utils.common_imports import *

train_logger = logging.getLogger("train_logger")


class ReinforceTrainerWRS:
    """
    REINFORCE trainer with reward shaping.
    To be used in a multi-agent context.
    Generalizes to single-agent case if used carefully.
    """

    # TODO: add GAE
    # TODO: add value function

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler,
        config: RtConfig,
    ):
        # TODO: add lr scheduler to accelerator
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left"  # needed for flash attention
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.config = config
        self.accelerator = Accelerator()
        self.model, self.optimizer = self.accelerator.prepare(
            model, 
            optimizer)
        self.tally = RtTally(tokenizer=tokenizer)

    def advantages_to_aa_scores(
        self,
        a1: np.ndarray,
        a2: np.ndarray,
    ):
        """
        Calculate the advantage alignment scores with vectorization.

        Args:
            a1 (np.ndarray):
                The first advantage array.
            a2 (np.ndarray):
                The second advantage array.
            gamma (float, optional):
                The discount factor. Defaults to 0.9.
            beta (float, optional):
                The shaping factor. Defaults to 1.0.
            regulate_var (bool, optional):
                Whether to regulate variance. Defaults to False.
            time_decay (bool, optional):
            Whether to apply 1/t regularization. Defaults to False.

        Returns:
            adv_align_terms (np.ndarray): The advantage alignment terms.

        The advantage alignment score is calculated as:
        .. math::
            A^*(s_t, a_t, b_t) = A^1(s_t, a_t, b_t) + \\beta \\gamma \\cdot
            \\left( \\sum_{k < t} \\gamma^{t-k} A^1(s_k, a_k, b_k) \\right)
            A^2(s_t, a_t, b_t)

        Refer to https://arxiv.org/abs/2406.14662

        """
        if len(a1.shape) == 1:
            a1 = a1[None, :]
        if len(a2.shape) == 1:
            a2 = a2[None, :]
        a1 = np.array(a1)
        a2 = np.array(a2)
        gamma = self.config.discount_factor
        beta = self.config.ad_align_beta

        # Regular alignment terms
        T = a1.shape[1]
        discounted_a1 = a1 * (gamma * np.ones(shape=(1, T))) ** (-np.arange(0, T, 1))
        discounted_sums_a1 = discounted_a1 @ (np.triu(np.ones((T, T))) - np.identity(T))
        t_discounts = (gamma * np.ones(shape=(1, T))) ** (np.arange(0, T, 1))
        alignment_terms = gamma * t_discounts * discounted_sums_a1 * a2

        self.tally.add_metric(
            path=["advantage_alignment_terms"], metric=alignment_terms
        )

        # Normalize alignment terms (across same time step)
        if self.config.use_variance_regularization_in_ad_align:
            reg_coef = np.std(a1[:, -1]) / (np.std(alignment_terms[:, -1]) + 1e-10)
            alignment_terms = reg_coef * alignment_terms

        # 1/1+t Regularization
        if self.config.use_time_regularization_in_ad_align:
            t_values = np.arange(1, T + 1)
            alignment_terms = alignment_terms / t_values

        self.tally.add_metric(
            path=["normalized_advantage_alignment_terms"], 
            metric=alignment_terms
        )

        adv_align_terms = a1 + beta * alignment_terms

        return adv_align_terms.squeeze()

    def discount_returns(self, rewards: np.ndarray):
        """
        TODO: docstring
        """
        # TODO fix
        dr = np.zeros(rewards.shape)
        T = rewards.shape[0]
        dr[-1] = rewards[-1]
        for i in range(T - 2, -1, -1):
            dr[i] = rewards[i] + self.config.discount_factor * dr[i + 1]
        return dr

    def get_response_scores_from_data(self, paths: list[str]):
        """
        This is where the reward shaping occurs.

        Process all of the rewards in the data at once.
        Obtains the score attributed to each of the agent's responses.
        #TODO: docstring
        Args:

        Returns:
            all_response_scores : list[torch.Tensor]
        """
        all_response_scores = {}

        for filepath in paths:
            rewards = []
            co_rewards = []
            # Extract the rewards from conversation history
            # Each response corresponds to one component
            # of the score vector
            with open(filepath) as f:
                conversation = json.load(f)
                for message in conversation:
                    r = message.get("reward", None)
                    co_r = message.get("co_reward", None)
                    if r != None:
                        rewards.append(r)
                        co_rewards.append(co_r)

            s = np.array(rewards)
            co_s = np.array(co_rewards)

            self.tally.add_metric(path=["rewards"], metric=s)

            self.tally.add_metric(path=["co_rewards"], metric=co_s)

            if self.config.use_sum_rewards:
                s = s + co_s
                co_s = s

            s = self.discount_returns(rewards=s)
            co_s = self.discount_returns(rewards=co_s)

            self.tally.add_metric(path=["discounted_returns"], metric=s)

            self.tally.add_metric(path=["co_discounted_returns"], metric=co_s)

            if self.config.use_advantage_alignment:
                s = self.advantages_to_aa_scores(a1=s, a2=co_s)

            all_response_scores[filepath] = torch.Tensor(s)

        return all_response_scores

    def get_training_data(self, paths: list[str]):
        """
        TODO: docstring
        Converts external folder of json conversation
        files into lists of training torch tensors.
        """
        all_contexts = []
        all_scores = []
        all_action_masks = []

        all_per_response_scores = self.get_response_scores_from_data(paths=paths)

        for filepath in paths:
            # Load conversation from json file
            with open(filepath) as f:
                conversation = json.load(f)
            formatted_conversation = self.tokenizer.apply_chat_template(
                conversation,
                add_generation_prompt=False,
                tokenize=False,
                use_system_prompt=True,
            )
            context = self.tokenizer.encode(
                formatted_conversation, 
                return_tensors="pt", 
                add_special_tokens=False
            ).squeeze()

            scores = torch.zeros(context.shape)
            action_mask = torch.zeros(context.shape)

            per_response_scores = all_per_response_scores[filepath]

            response_score_pos = 0
            for i, message in enumerate(conversation):
                # TODO: verify that EOS token is properly included
                role = message.get("role", None)
                if role != "assistant":
                    continue
                response = message.get("content", None)
                score = per_response_scores[response_score_pos]
                response_score_pos += 1
                nb_tokens_before_response = (
                    self.tokenizer.apply_chat_template(
                        conversation[:i],
                        add_generation_prompt=True,
                        tokenize=True,
                        return_tensors="pt",
                    )
                    .squeeze()
                    .shape[0]
                )
                # TODO: write this cleanly

                nb_tokens_in_response = len(
                    self.tokenizer.encode(response))
                a = nb_tokens_before_response
                b = nb_tokens_in_response
                scores[a : a + b + 1] = score
                action_mask[a : a + b + 1] = 1.0

            all_contexts.append(context)
            all_scores.append(scores)
            all_action_masks.append(action_mask)

        return all_contexts, all_scores, all_action_masks

    def get_expected_entropy(
        self,
        rollout_ids: list[str],
        contexts: torch.Tensor,
        shifted_contexts: torch.Tensor,
        logits: torch.Tensor,
        action_mask: torch.Tensor,
    ):
        """
        Computes entropy were actions are LLM-generated strings of tokens.
        """
        # TODO: check if this is right,
        # Not simple with actions being strings
        # import pdb; pdb.set_trace()
        try:
            B, S, T = logits.shape
        except:
            print("Missing batch dimension.")

        # TODO: check infs here (special.xlogy)
        token_entropy_terms = -F.softmax(logits, dim=-1) \
             * F.log_softmax(logits, dim=-1)
        # (B, S, T)

        # Log entropy terms for each token generated
        generated_tokens_entr = torch.gather(
            input=token_entropy_terms, 
            index=shifted_contexts[:, :, None].long(),
            dim=-1, 
        )
        self.tally.add_contextualized_token_metrics(
            rollout_ids=rollout_ids,
            metric_id="token_entropy_terms",
            contexts=contexts,
            metrics=generated_tokens_entr.squeeze(),
            action_mask=action_mask,
        )
        expected_entropy = token_entropy_terms.sum()
        return expected_entropy

    def get_kl_divergence(
        self,
        input_ids,
        attention_mask,
        action_log_probs,
        index,
    ):
        """
        TODO
        # Ref 1: http://joschu.net/blog/kl-approx.html
        # Ref 2: https://github.dev/huggingface/trl/
        # blob/main/trl/trainer/grpo_trainer.py#L945
        """

        self.model.eval()

        # TODO: (Dereck) Complete this code

        # Disable policy adapter to run inference on base model
        with torch.no_grad():
            with self.model.disable_adapter():
                ref_model_logits = self.model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask
                )[0]
        self.model.train()
        ref_model_logits = ref_model_logits / self.config.temperature
        # (B, S, V)
        ref_model_log_probs = F.log_softmax(ref_model_logits, dim=-1)
        # (B, S, V)
        ref_model_action_log_probs = ref_model_log_probs.gather(
            dim=-1, index=index.unsqueeze(-1)
        ).squeeze(
            -1
        )  # (B,S)

        # Approximating KL Divergence (see refs in docstring)
        kl_div = (
            torch.exp(ref_model_action_log_probs - action_log_probs)
            - (ref_model_action_log_probs - action_log_probs)
            - 1
        ).sum()

        return kl_div

    def mask_non_restricted_token_logits(
        self, 
        logits: torch.Tensor):
        """
        TODO: docstring
        """
        # TODO: verify
        allowed_token_ids = []
        for token in self.config.restrict_tokens:
            token_ids = self.tokenizer(
                token, 
                add_special_tokens=False)["input_ids"]
            allowed_token_ids.append(token_ids[0])
        allowed_token_ids = torch.tensor(
            allowed_token_ids, 
            device=logits.device)
        # Mask log_probs and probs to only allowed tokens
        mask = torch.zeros_like(logits).bool()  # (B, S, V)
        mask[..., allowed_token_ids] = True
        logits = torch.where(
            mask,
            logits,
            torch.tensor(float("-inf"), device=logits.device),
        )
        return logits

    def apply_reinforce_step(
        self,
        rollout_ids: list[str],
        contexts: list[torch.Tensor],
        scores: list[torch.Tensor],
        action_masks: list[torch.Tensor],
    ):
        """
        TODO: docstring

        See https://huggingface.co/docs/accelerate/usage_guides/
        gradient_accumulation#converting-it-to-accelerate
        """
        self.model.train()
        loss = 0.0
        mb_size = self.config.mini_batch_size
        device = self.accelerator.device
        self.tokenizer.padding_side = "left"

        nb_rollouts = len(contexts)
        self.tally.add_metric(
            path=["nb_rollouts"],
            metric=nb_rollouts)

        # Count total number of tokens trained on
        total_nb_action_tokens = 0
        for am in action_masks:
            nb_tokens = am.shape[0]
            self.tally.add_metric(
                path=["nb_tokens", "action_tokens"],
                metric=nb_tokens)
            total_nb_action_tokens += nb_tokens


        for mb in range(0, len(contexts), mb_size):

            rollout_ids_mb = rollout_ids[mb:mb+mb_size]

            # Convert sequences to padded tensor minibatches
            contexts_mb = [c[:-1] for c in contexts[mb : mb + mb_size]]
            tok_out = self.tokenizer.pad(
                {"input_ids": contexts_mb}, 
                padding="longest", 
                return_tensors="pt"
            )
            contexts_mb = tok_out.input_ids.to(device)
            attention_mask = tok_out.attention_mask.to(device)
            shifted_contexts_mb = [c[1:] for c in contexts[mb : mb + mb_size]]
            tok_out = self.tokenizer.pad(
                {"input_ids": shifted_contexts_mb},
                padding="longest",
                return_tensors="pt",
            )
            shifted_contexts_mb = tok_out.input_ids.to(device)
            scores_mb = [s[1:] for s in scores[mb : mb + mb_size]]
            scores_mb = (
                pad_sequence(
                    sequences=scores_mb,
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


            # Forward pass
            logits = self.model(
                input_ids=contexts_mb, 
                attention_mask=attention_mask)[
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

            self.tally.add_contextualized_token_metrics(
                rollout_ids=rollout_ids_mb,
                metric_id="next_token_probs",
                contexts=contexts_mb,
                metrics=torch.exp(action_log_probs),
                action_mask=action_mask_mb,
            )

            rewarded_action_log_probs = (
                action_mask_mb 
                * scores_mb 
                * action_log_probs)
            # (B, S)

            # Add value term to loss
            nb_act_tokens = torch.sum(action_mask_mb)
            mb_value = -rewarded_action_log_probs.sum()
            self.tally.add_metric(
                path=["loss_mb_total", "value_mb_total"],
                metric=mb_value.item())
            loss += mb_value

            # Add entropy regularization term to loss
            if self.config.entropy_coeff != 0.0:
                mb_entropy = self.get_expected_entropy(
                    rollout_ids=rollout_ids_mb,
                    contexts=contexts_mb,
                    shifted_contexts=shifted_contexts_mb,
                    logits=logits,
                    action_mask=action_mask_mb,
                )
                ent = self.config.entropy_coeff * mb_entropy
                self.tally.add_metric(
                    path=["loss_mb_total", "entropy_mb_total"],
                    metric=ent.item()
                )
                loss += ent

            # Add KL-divergence regularization term to loss
            if self.config.kl_coeff != 0.0:
                # TODO: verify
                mb_kl = self.get_kl_divergence(
                    input_ids=contexts_mb,
                    attention_mask=attention_mask,
                    action_log_probs=action_log_probs,
                    index=shifted_contexts_mb,
                )
                self.tally.add_metric(
                    path=["mb_kl_loss_terms"], 
                    metric=mb_kl.item())
                loss += self.config.kl_coeff * mb_kl

            # Normalize over number tokens generated
            loss /= total_nb_action_tokens

            # Accumulate gradient
            self.accelerator.backward(loss)

            loss = 0.0

            # ensure gpu memory is freed
            del log_probs
            del contexts_mb
            del shifted_contexts_mb
            del scores_mb
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
                self.model.parameters(), 
                self.config.gradient_clipping
            )
            # TODO: log at right place
            self.tally.add_metric(
                path=["gradient_norm"], 
                metric=grad_norm.item())

        # Log gradient norms
        grad_l_norms = []
        for name, p in self.model.named_parameters():
            if p.grad is None: continue
            grad_norm = p.grad.data.norm(2).item()
            grad_l_norms.append(grad_norm)
        self.tally.add_metric(
            path=["gradient_norms", "layer_norms"],
            metric=np.array(grad_l_norms)
            )



        # Take step
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Clear
        # TODO: verify
        self.accelerator.clear(self.model, self.optimizer)
        import gc

        gc.collect()
        torch.cuda.empty_cache()

    def apply_reinforce_step_on_paths(self, paths: list[str]):
        """
        TODO: docstring
        """
        (contexts, 
        scores, 
        action_masks) = self.get_training_data(paths=paths)

        self.apply_reinforce_step(
            rollout_ids=paths,
            contexts=contexts, 
            scores=scores, 
            action_masks=action_masks
        )

    def export_training_metrics(self):
        """
        TODO: docstring
        """
        
        self.tally.save(path=self.config.logging_path)
