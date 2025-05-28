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
from utils.common_imports import *

train_logger = logging.getLogger("train_logger")


class ReinforceTrainer:
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler,
        config: RtConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.config = config
        self.accelerator = Accelerator()
        self.model, self.optimizer = self.accelerator.prepare(model, optimizer)

    def get_expected_entropy(self, logits: torch.Tensor, action_mask: torch.Tensor):
        """
        Computes entropy were actions are LLM-generated strings of tokens.
        """
        # TODO: check if this is right,
        # Not simple with actions being strings
        try:
            B, S, T = logits.shape
        except:
            print("Missing batch dimension.")
        entropy = -torch.special.xlogy(
            input=F.log_softmax(logits, dim=-1), other=F.softmax(logits, dim=-1)
        ).sum()
        return entropy / B

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
        # Ref 2: https://github.dev/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py#L945
        """

        self.model.eval()

        # TODO: (Dereck) Complete this code

        # Disable policy adapter to run inference on base model
        with torch.no_grad():
            with self.model.disable_adapter():
                ref_model_logits = self.model(
                    input_ids=input_ids, attention_mask=attention_mask
                )[0]
        self.model.train()
        ref_model_logits = ref_model_logits / self.config.temperature  # (B, S, V)
        ref_model_log_probs = F.log_softmax(ref_model_logits, dim=-1)  # (B, S, V)
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

    def mask_non_restricted_token_logits(self, logits: torch.Tensor):
        """
        TODO: docstring
        """
        allowed_token_ids = []
        for token in self.config.restrict_tokens:
            token_ids = self.tokenizer(token, add_special_tokens=False)["input_ids"]
            allowed_token_ids.append(token_ids[0])
        allowed_token_ids = torch.tensor(allowed_token_ids, device=logits.device)
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
        contexts: list[int],
        scores: list[int],
        action_masks: list[int],
    ):
        """
        TODO: docstring

        See https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation#converting-it-to-accelerate
        """
        step_metrics = {"mb_values": [], "mb_entropies": [], "mb_kl_terms": []}
        self.model.train()
        loss = 0.0
        mb_size = self.config.mini_batch_size
        device = self.accelerator.device
        nb_rollouts = len(contexts)
        gradient_accumulation_steps = nb_rollouts / mb_size

        for mb in range(0, len(contexts), mb_size):
            # Convert sequences to padded tensor minibatches

            contexts_mb = [c[:-1] for c in contexts[mb : mb + mb_size]]
            contexts_mb = (
                pad_sequence(
                    sequences=contexts_mb,
                    padding_value=self.tokenizer.pad_token_id,
                    batch_first=True,
                )
                .long()
                .to(device)
            )

            shifted_contexts_mb = [c[1:] for c in contexts[mb : mb + mb_size]]
            shifted_contexts_mb = (
                pad_sequence(
                    sequences=shifted_contexts_mb,
                    padding_value=self.tokenizer.pad_token_id,
                    batch_first=True,
                )
                .long()
                .to(device)
            )

            scores_mb = [s[1:] for s in scores[mb : mb + mb_size]]
            scores_mb = (
                pad_sequence(
                    sequences=scores_mb,
                    padding_value=self.tokenizer.pad_token_id,
                    batch_first=True,
                )
                .float()
                .to(device)
            )

            action_mask_mb = [am[1:] for am in action_masks[mb : mb + mb_size]]
            action_mask_mb = (
                pad_sequence(
                    sequences=action_mask_mb, padding_value=0.0, batch_first=True
                )
                .float()
                .to(device)
            )

            # Create attention mask to ignore padding tokens
            attention_mask = (
                (contexts_mb != self.tokenizer.pad_token_id).long().to(device)
            )  # (B, S)

            # Forward pass
            print(contexts_mb.shape)
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

            rewarded_action_log_probs = torch.special.xlogy(
                input=scores_mb * action_mask_mb, other=action_log_probs
            )  # (B, S)

            # Add value term to loss
            mb_value = -rewarded_action_log_probs.sum() / mb_size
            step_metrics["mb_values"].append(mb_value.item())
            loss += mb_value

            # Add entropy regularization term to loss
            if self.config.entropy_coeff != 0.0:
                mb_entropy = self.get_expected_entropy(
                    logits=logits, action_mask=action_mask_mb
                )
                step_metrics["mb_entropies"].append(mb_entropy.item())
                loss += self.config.entropy_coeff * mb_entropy

            # Add KL-divergence regularization term to loss
            if self.config.kl_coeff != 0.0:
                # TODO: verify
                mb_kl = self.get_kl_divergence(
                    input_ids=contexts_mb,
                    attention_mask=attention_mask,
                    action_log_probs=action_log_probs,
                    index=shifted_contexts_mb,
                )
                step_metrics["mb_kl_terms"].append(mb_kl.item())
                loss += self.config.kl_coeff * mb_kl

            del log_probs

            loss /= gradient_accumulation_steps
            self.accelerator.backward(loss)

            # ensure gpu memory is freed
            del contexts_mb
            del shifted_contexts_mb
            del scores_mb
            del action_mask_mb
            del attention_mask
            del logits
            del loss
            del action_log_probs
            del rewarded_action_log_probs
            torch.cuda.empty_cache()

        # self.accelerator.sync_gradients

        # Clip gradients and take step
        if self.config.gradient_clipping is not None:
            grad_norm = self.accelerator.clip_grad_norm_(
                self.model.parameters(), self.config.gradient_clipping
            )
            step_metrics["gradient_norm"] = grad_norm.item()

        # Take step
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Clear
        # TODO: verify
        self.accelerator.clear(self.model, self.optimizer)
        import gc

        gc.collect()
        torch.cuda.empty_cache()
