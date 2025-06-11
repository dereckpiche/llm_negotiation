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
from training.tokenizer_action_masking import get_assistant_actions_mask_and_score
from utils.common_imports import *
from utils.time_and_memory_utils import *
from utils.print_logger import *




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
        lr_scheduler: torch.optim.lr_scheduler,
        config: RtConfig,
    ):
        # TODO: add lr scheduler to accelerator
        model.train()
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

        self.logger = PrintLogger(
            logging.getLogger("reinforcer_trainer_logger"))

        if self.config.use_gradient_checkpointing == True:
            self.logger.info("Enabling gradient checkpointing.")
            self.model.gradient_checkpointing_enable(dict(use_reentrant=False))


        # self.tally.add_metric(
        #     path=["tokenizer_eot_id"],
        #     metric=self.tokenizer.eos_token_id)
        # TODO
        # log number of trainable parameters
        # log data type of model
        # log adapter type, rank, etc.
        # log optimizer learning rate
        # log model data type 
        # log adapter data type


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

        # Normalize advantages
        if self.config.ad_align_normalize_advantages:
            self.tally.add_metric(
                path=["a1_before_normalizing"],
                metric=a1
            )
            a1 = (a1 - np.mean(a1)) / (np.std(a1) + 1e-6)
            self.tally.add_metric(
                path=["a1_after_normalizing"],
                metric=a1
            )
            self.tally.add_metric(
                path=["a2_before_normalizing"],
                metric=a2
            )
            a2 = (a2 - np.mean(a2)) / (np.std(a2) + 1e-6)
            self.tally.add_metric(
                path=["a2_after_normalizing"],
                metric=a2
            )

        # Regular alignment terms
        T = a1.shape[1]
        discounted_a1 = a1 * (gamma * np.ones(shape=(1, T))) ** (-np.arange(0, T, 1))
        discounted_sums_a1 = discounted_a1 @ (np.triu(np.ones((T, T))) - np.identity(T))
        t_discounts = (gamma * np.ones(shape=(1, T))) ** (np.arange(0, T, 1))
        adalign_weights = beta * gamma * t_discounts * discounted_sums_a1 

        self.tally.add_metric(
            path=["raw_advantage_alignment_weights"],
            metric=adalign_weights
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
                metric=positive_signs.sum() / adalign_weights.size
            )
            self.tally.add_metric(
                path=["adalign_weights_ratio_negative_signs"], 
                metric=negative_signs.sum() / adalign_weights.size
            )
            # (rest are 0)

            self.tally.add_metric(
                path=["ad_align_weights_after_using_sign"], 
                metric=adalign_weights
            )

        # Use clipping 
        if self.config.ad_align_clipping not in [0.0, None]:
            
            upper_mask = adalign_weights > 1
            lower_mask = adalign_weights < -1

            adalign_weights = np.clip(
                adalign_weights, 
                -self.config.ad_align_clipping, 
                self.config.ad_align_clipping
            )
            clipping_ratio = (np.sum(upper_mask) + np.sum(lower_mask)) / upper_mask.size

            self.tally.add_metric(
                path=["ad_align_clipping_ratio"], 
                metric=clipping_ratio
            )

            self.tally.add_metric(
                path=["ad_align_weights_after_clipping"], 
                metric=adalign_weights
            )

        
        # 1/1+t Regularization
        if self.config.use_time_regularization_in_ad_align:
            t_values = np.arange(1, T + 1)
            adalign_weights = adalign_weights / t_values
            self.tally.add_metric(
                path=["adalign_weights_after_1_over_t_reg"], 
                metric=adalign_weights
            )

        # Use coop on t=0
        if self.config.ad_align_force_coop_first_step:
            adalign_weights[:, 0] = 1
            self.tally.add_metric(
                path=["adalign_weights_after_force_coop_first_step"], 
                metric=adalign_weights
            )

        opp_shaping_terms = adalign_weights * a2

        self.tally.add_metric(
            path=["ad_align_opp_shaping_terms"], 
            metric=opp_shaping_terms
        )

        # Normalize alignment terms (across same time step)
        if self.config.use_variance_regularization_in_ad_align:
            # TODO: verify
            reg_coef = np.std(a1[:, -1]) / (np.std(opp_shaping_terms[:, -1]) + 1e-9)
            opp_shaping_terms *= reg_coef
            self.tally.add_metric(
                path=["opp_shaping_terms_after_var_reg"], 
                metric=opp_shaping_terms
            )

        ad_align_scores = a1 + opp_shaping_terms

        self.tally.add_metric(
            path=["final_advantage_alignment_scores"], 
            metric=ad_align_scores
        )

        self.logger.info(f"\n \n After AdAlign \n  {ram_usage()} \n {vram_usage()}")


        return ad_align_scores.squeeze()

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

            # Apply chat template, get token ids
            formatted_conversation = self.tokenizer.apply_chat_template(
                conversation,
                add_generation_prompt=False,
                tokenize=False,
                use_system_prompt=True,
            )
            tokens = self.tokenizer.encode(
                formatted_conversation,  
                return_tensors="pt", 
                add_special_tokens=True
            ).squeeze(0).long()

            per_response_scores = all_per_response_scores[filepath]
            scores, action_mask = get_assistant_actions_mask_and_score(
                tokenizer=self.tokenizer,
                assistant_msg_scores=torch.Tensor(per_response_scores).squeeze(),
                token_ids = tokens
            )
            # decoded = self.tokenizer.convert_ids_to_tokens(tokens.tolist())
            # df = {"Tokens": decoded, "Score": scores, "Action Mask": action_mask}
            # df = pd.DataFrame(df)
            # print(df.to_string())

            assert tokens.shape == scores.shape == action_mask.shape

            all_contexts.append(tokens)
            all_scores.append(scores)
            all_action_masks.append(action_mask)

        return all_contexts, all_scores, all_action_masks

    def mask_non_restricted_token_logits(
        self, 
        logits: torch.Tensor):
        """
        TODO: docstring
        """
        # TODO: verify. Not sure what we do here is differentiable
        # also, we recompute for nothing

        if self.config.restrict_tokens is not None:
            allowed_token_ids = []
            for token in self.config.restrict_tokens:
                token_ids = self.tokenizer(
                    token, 
                    add_special_tokens=False)["input_ids"]
                allowed_token_ids.append(token_ids[0])
            allowed_token_ids.append(self.tokenizer.eos_token_id) # This token should always be active   
            allowed_token_ids = torch.tensor(
                allowed_token_ids, 
                device=logits.device)
            # Mask log_probs and probs to only allowed tokens
            mask = torch.zeros_like(logits).bool()  # (B, S, V)
            mask[..., allowed_token_ids] = True
            logits = torch.where(
                mask,
                logits,
                torch.tensor(-float('inf'), device=logits.device),
            )

        return logits

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
        try:
            B, S, T = logits.shape
        except:
            print("Missing batch dimension.")


        token_entropy_terms = -F.softmax(logits, dim=-1) \
             * F.log_softmax(logits, dim=-1)
        # (B, S, T)

        # We only take the entropy of actions
        token_entropy_terms *= action_mask[:, :, None]

        if self.config.log_ctz_entropy:
            self.tally.add_contextualized_token_metrics(
                rollout_ids=rollout_ids,
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
        rollout_ids: list[str],
        contexts: torch.Tensor,
        shifted_contexts: torch.Tensor,
        attention_mask: torch.Tensor,
        action_log_probs: torch.Tensor,
        action_mask: torch.Tensor
    ):
        """
        TODO: docstring
        # Ref 1: http://joschu.net/blog/kl-approx.html
        # Ref 2: https://github.dev/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py#L1332
        """

        # Disable policy adapter to run inference on base model
        with torch.no_grad():
            with self.model.disable_adapter():
                ref_model_logits = self.model(
                    input_ids=contexts, 
                    attention_mask=attention_mask
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
                rollout_ids=rollout_ids,
                metric_id="kl",
                contexts=shifted_contexts,
                metrics=kl_div,
                action_mask=action_mask,
            )

        # We only care about KLD of action tokens
        kl_div *= action_mask

        kl_div = kl_div.sum()

        return kl_div



    def get_gradient_magnitude(self, loss_term: torch.Tensor):
        """
        TODO: docstring
        """
        with torch.no_grad():
            grads = torch.autograd.grad(
                loss_term,
                [p for p in self.model.parameters() if p.requires_grad],
                retain_graph=True, 
                allow_unused=True               
            )
            grads = [g for g in grads if g is not None]
            if not grads:                      
                return torch.tensor(0.0, device=loss_term.device)
            return torch.norm(torch.stack([g.norm(2) for g in grads])).item()

    def apply_reinforce_step(
        self,
        rollout_ids: list[str],
        contexts: list[torch.Tensor],
        scores: list[torch.Tensor],
        action_masks: list[torch.Tensor],
    ):
        """
        TODO: docstring

        REINFORCE gradient estimators are a sum of these terms:
            s(a, s) ∇ log π(a|s)
        In this code, "s" is called "score". 

        See https://huggingface.co/docs/accelerate/usage_guides/
        gradient_accumulation#converting-it-to-accelerate
        """

        self.logger.info(f"\n Before Reinforce Step \n  {ram_usage()} \n {vram_usage()}")

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
        
        self.tally.add_metric(
            path=["nb_tokens", "batch_action_tokens"],
            metric=total_nb_action_tokens)


        for mb in range(0, len(contexts), mb_size):
            rollout_ids_mb = rollout_ids[mb:mb+mb_size]

            # Convert sequences to padded tensor minibatches
            tokens_mb = contexts[mb : mb + mb_size]
            tok_out = self.tokenizer.pad(
                {"input_ids": tokens_mb}, 
                padding="longest", 
                return_tensors="pt"
            )
            tokens_mb = tok_out.input_ids.to(device)
            attention_mask = tok_out.attention_mask.to(device)[:, :-1]
            contexts_mb = tokens_mb[:, :-1]
            shifted_contexts_mb= tokens_mb[:, 1:]

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

            if self.config.log_ctz_next_token_score:
                self.tally.add_contextualized_token_metrics(
                    rollout_ids=rollout_ids_mb,
                    metric_id="next_token_score",
                    contexts=shifted_contexts_mb,
                    metrics=scores_mb,
                    action_mask=action_mask_mb,
                )

            # Forward pass + cast to FP-32 for higher prec.
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

            if self.config.log_ctz_next_token_log_prob:
                self.tally.add_contextualized_token_metrics(
                    rollout_ids=rollout_ids_mb,
                    metric_id="next_token_log_prob",
                    contexts=shifted_contexts_mb,
                    metrics=action_log_probs,
                    action_mask=action_mask_mb,
                )
            
            if self.config.log_ctz_next_token_prob:
                self.tally.add_contextualized_token_metrics(
                    rollout_ids=rollout_ids_mb,
                    metric_id="next_token_prob",
                    contexts=shifted_contexts_mb,
                    metrics=torch.exp(action_log_probs),
                    action_mask=action_mask_mb,
                )

            if self.config.log_ctz_top_k != 0:
                # Log top K ids and probs

                self.logger.info(f"\n Before Logging Top K \n  {ram_usage()} \n {vram_usage()}")

                top_k_indices = torch.topk(logits, 
                k=self.config.log_ctz_top_k, dim=-1).indices

                # import pdb; pdb.set_trace()
                if self.config.log_ctz_top_k_tids:
                    self.tally.add_contextualized_token_metrics(
                        rollout_ids=rollout_ids_mb,
                        metric_id=f"top_{self.config.log_ctz_top_k}_tids",
                        contexts=shifted_contexts_mb,
                        metrics=top_k_indices,
                        action_mask=action_mask_mb,
                        to_tids=True
                    )
                if self.config.log_ctz_top_k_probs:
                    self.tally.add_contextualized_token_metrics(
                        rollout_ids=rollout_ids_mb,
                        metric_id=f"top_{self.config.log_ctz_top_k}_probs",
                        contexts=shifted_contexts_mb,
                        metrics=torch.exp(log_probs).gather(
                            dim=-1, 
                            index=top_k_indices),
                        action_mask=action_mask_mb,
                    )

                self.logger.info(f"\n After Logging Top K \n  {ram_usage()} \n {vram_usage()}")

            

            rewarded_action_log_probs = (
                action_mask_mb 
                * scores_mb 
                * action_log_probs)
            # (B, S)
            if self.config.log_ctz_top_slogpi:
                self.tally.add_contextualized_token_metrics(
                    rollout_ids=rollout_ids_mb,
                    metric_id="next_token_slogpi",
                    contexts=shifted_contexts_mb,
                    metrics=rewarded_action_log_probs,
                    action_mask=action_mask_mb,
                )

            # Add value term to loss
            nb_act_tokens = torch.sum(action_mask_mb)
            mb_value = -rewarded_action_log_probs.sum()
            self.tally.add_metric(
                path=["loss_mb_total", "value_mb_total"],
                metric=mb_value.item())

            if self.config.log_value_gradient_terms:
                self.tally.add_metric(
                    path=["gradient_term_magnitudes", "value"],
                    metric=self.get_gradient_magnitude(
                        loss_term=mb_value
                    )
                )
            loss += mb_value

            # Add entropy regularization term to loss
            if self.config.entropy_coeff != 0.0:

                self.logger.info(f"\n Before Computing Entropy \n  {ram_usage()} \n {vram_usage()}")

                mb_entropy = self.get_expected_entropy(
                    rollout_ids=rollout_ids_mb,
                    contexts=contexts_mb,
                    shifted_contexts=shifted_contexts_mb,
                    logits=logits,
                    action_mask=action_mask_mb,
                )
                mb_entropy *= self.config.entropy_coeff 
                self.tally.add_metric(
                    path=["loss_mb_total", "entropy_mb_total"],
                    metric=mb_entropy.item()
                )
                
                if self.config.log_entropy_gradient_terms:
                    self.tally.add_metric(
                        path=["gradient_term_magnitudes", "entropy"],
                        metric=self.get_gradient_magnitude(
                            loss_term=mb_entropy
                        )
                    )
                loss += mb_entropy

                self.logger.info(f"\n After Computing Entropy \n  {ram_usage()} \n {vram_usage()}")

            # Add KL-divergence regularization term to loss
            if self.config.kl_coeff != 0.0:

                self.logger.info(f"\n Before Computing KLD \n  {ram_usage()} \n {vram_usage()}")

                # TODO: verify
                mb_kl = self.get_kl_divergence(
                    rollout_ids=rollout_ids_mb,
                    contexts=contexts_mb,
                    shifted_contexts=shifted_contexts_mb,
                    attention_mask=attention_mask,
                    action_log_probs=action_log_probs,
                    action_mask=action_mask_mb
                )
                mb_kl *= self.config.kl_coeff

                self.tally.add_metric(
                    path=["mb_kl_loss_terms"], 
                    metric=mb_kl.item())

                if self.config.log_kl_gradient_terms:
                    self.tally.add_metric(
                        path=["gradient_term_magnitudes", "kl"],
                        metric=self.get_gradient_magnitude(
                            loss_term=mb_kl
                        )
                    )

                loss += mb_kl

                self.logger.info(f"\n After Computing KLD \n  {ram_usage()} \n {vram_usage()}")


            # Normalize over number tokens generated
            # loss /= total_nb_action_tokens

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
