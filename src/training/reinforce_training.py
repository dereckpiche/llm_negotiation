import logging
import os
import random
from collections import defaultdict
from contextlib import contextmanager, nullcontext

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.nn.utils.rnn import pad_sequence

from utils.common_imports import *

compute_logger = logging.getLogger("compute_logger")
memory_logger = logging.getLogger("memory_logger")
model_logger = logging.getLogger("model_logger")


def reinforce_train(
    model,
    contexts_list,
    scores_list,
    output_masks_list,
    optimizer,
    nb_epochs=1,
    mb_size=1,
    mb_per_step=-1,
    output_path=None,
    tokenizer=None,
    gradient_checkpointing=False,
    entropy_coef=0,
    kl_loss_coef=0,
    temperature=1.0,  # new hyperparameter to control softmax temperature during training
    use_accelerate_gradaccum=False,
):
    """
    Args:
        model (torch.nn.Module): The language model with a value head to be optimized.
        ref_model (torch.nn.Module): Reference model used for KL penalty.
        contexts_list (list of torch.Tensor): List of input contexts, each of shape (S, V).
        scores_list (list of torch.Tensor): List of estimated scores for each time step, each of shape (S,).
        output_masks_list (list of torch.Tensor): List of masks for output tokens, each of shape (S,).
        optimizer (torch.optim.Optimizer, optional): Optimizer for training the model. If None, a default optimizer will be created.
        nb_epochs (int): Number of epochs to train over the dataset.
        mb_size (int): Minibatch size, the number of sequences processed at once.
        mb_per_step (int): Number of minibatches to accumulate gradients over before taking an optimizer step.
        clip_param (float, optional): Clipping parameter epsilon for PPO, default is 0.2.
        vf_coef (float, optional): Coefficient for value loss, default is 0.5.
        entropy_coef (float, optional): Coefficient for entropy bonus, default is 0.01.
        temperature (float): Hyperparameter to control the softmax temperature. Must be > 0 (default: 1.0).

    scores:
        float: The total loss value for the training step.
    """
    model.train()
    if gradient_checkpointing == True:
        model.gradient_checkpointing_enable(dict(use_reentrant=False))

    # if output_path:
    #     output_train_data_debug(output_path,
    #                             contexts_list,
    #                             scores_list,
    #                             output_masks_list,
    #                             tokenizer)

    if optimizer is None:
        raise ValueError(
            "Optimizer must be provided. Please pass an optimizer instance."
        )

    verify_reinforce_train_inputs(contexts_list, scores_list, output_masks_list)

    # Calculate the maximum context length in terms of number of tokens
    max_context_length = max(context.size(0) for context in contexts_list)
    model_logger.info(f"Max context length (in tokens): {max_context_length}")

    max_memory_usage = 0  # Initialize max memory usage

    # nb_trajectories_we_train_on is now the multiple of mb_size
    nb_trajectories_we_train_on = len(contexts_list) - len(contexts_list) % mb_size
    if mb_per_step == -1:
        gradient_accumulation_steps = nb_trajectories_we_train_on // mb_size
    else:
        gradient_accumulation_steps = mb_per_step

    if use_accelerate_gradaccum:
        # Initialize the accelerators (https://huggingface.co/docs/accelerate/en/usage_guides/gradient_accumulation)
        model_accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps
        )
    else:
        model_accelerator = Accelerator()
    model, optimizer = model_accelerator.prepare(model, optimizer)

    loss_dict = defaultdict(list)
    for epoch in range(nb_epochs):
        for i in range(0, len(contexts_list), mb_size):
            if use_accelerate_gradaccum:
                context_manager = model_accelerator.accumulate(model)
            else:
                context_manager = nullcontext()
            with context_manager:
                # Get the minibatch
                context_batch = contexts_list[i : i + mb_size]
                return_batch = scores_list[i : i + mb_size]
                mask_batch = output_masks_list[i : i + mb_size]

                action_batch = [a[1:] for a in context_batch]
                return_batch = [r[1:] for r in return_batch]
                mask_batch = [m[1:] for m in mask_batch]
                context_batch = [c[:-1] for c in context_batch]

                # Pad sequences
                action_batch = pad_sequence(action_batch, batch_first=True).long()
                context_batch = pad_sequence(context_batch, batch_first=True).long()
                return_batch = pad_sequence(return_batch, batch_first=True).float()
                mask_batch = pad_sequence(mask_batch, batch_first=True).float()

                # Create attention mask to ignore padding tokens
                attention_mask = (
                    context_batch != 0
                ).long()  # context_batch: (B, S) -> attention_mask: (B, S)

                # Move data to the appropriate device
                action_batch = action_batch.to(model_accelerator.device)  # (B, S)
                context_batch = context_batch.to(model_accelerator.device)  # (B, S)
                return_batch = return_batch.to(model_accelerator.device)  # (B, S)
                mask_batch = mask_batch.to(model_accelerator.device)  # (B, S)
                attention_mask = attention_mask.to(model_accelerator.device)  # (B, S)

                # Forward pass
                outputs = model(input_ids=context_batch, attention_mask=attention_mask)

                logits = outputs[0]  # (B, S, V)

                # Apply temperature scaling before computing log probabilities
                assert temperature > 0, "Temperature must be greater than 0."

                logits = logits / temperature  # (B, S, V)

                # Compute new log probabilities
                log_probs = F.log_softmax(logits, dim=-1)  # (B, S, V)
                # Apply mask to log probabilities and values
                action_log_probs = log_probs.gather(
                    dim=-1, index=action_batch.unsqueeze(-1)
                ).squeeze(
                    -1
                )  # (B,S)

                entropy = -log_probs * F.softmax(logits, dim=-1)  # (B, S, V)
                entropy = entropy.sum(dim=-1)  # (B, S)

                kl_div = 0
                if kl_loss_coef != 0.0:
                    kl_div = compute_kl_div(
                        model,
                        input_ids=context_batch,
                        attention_mask=attention_mask,
                        action_log_probs=action_log_probs,
                        index=action_batch,
                        temperature=temperature,
                    )
                per_token_kl = kl_loss_coef * (kl_div * mask_batch)

                rewarded_action_log_probs = action_log_probs * (
                    return_batch * mask_batch
                ) + entropy_coef * (
                    entropy * mask_batch
                )  # (B,S)

                rewarded_action_log_probs = rewarded_action_log_probs - per_token_kl

                # Avoid division by zero by adding a small epsilon value to the denominator
                epsilon = 1e-8
                loss = -rewarded_action_log_probs.sum(dim=1) / (
                    torch.sum(mask_batch, dim=1) + epsilon
                )  # (B,)
                loss = loss.mean()
                loss_dict["loss"].append(loss.item())

                if use_accelerate_gradaccum:
                    model_accelerator.backward(loss)
                    optimizer.step()
                    optimizer.zero_grad()
                else:
                    loss = loss / gradient_accumulation_steps
                    model_accelerator.backward(loss)
                    if ((i + mb_size) // mb_size) % gradient_accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()

                # Update max GPU memory usage
                current_memory_usage = torch.cuda.max_memory_allocated(
                    model_accelerator.device
                )
                if current_memory_usage > max_memory_usage:
                    max_memory_usage = current_memory_usage

    # Log max GPU memory usage after training
    memory_logger.info(
        f"Max GPU memory usage during training: {max_memory_usage / (1024 ** 2):.2f} MB"
    )
    model_logger.info(f"loss: {np.mean(loss_dict['loss']):.4f}")
    model_accelerator.clear(model, optimizer)
    return {"loss": np.mean(loss_dict["loss"])}


def compute_kl_div(
    model, input_ids, attention_mask, action_log_probs, index, temperature
):
    model.eval()

    # Disable policy adapter to run inference on base model
    with torch.no_grad():
        with model.disable_adapter():
            ref_model_logits = model(
                input_ids=input_ids, attention_mask=attention_mask
            )[0]

    model.train()

    ref_model_logits = ref_model_logits / temperature  # (B, S, V)
    ref_model_log_probs = F.log_softmax(ref_model_logits, dim=-1)  # (B, S, V)
    ref_model_action_log_probs = ref_model_log_probs.gather(
        dim=-1, index=index.unsqueeze(-1)
    ).squeeze(
        -1
    )  # (B,S)

    # Approximating KL Divergence
    # Ref 1: http://joschu.net/blog/kl-approx.html
    # Ref 2: https://github.dev/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py#L945
    kl_div = (
        torch.exp(ref_model_action_log_probs - action_log_probs)
        - (ref_model_action_log_probs - action_log_probs)
        - 1
    )

    return kl_div


def verify_reinforce_train_inputs(contexts_list, scores_list, output_masks_list):
    """
    Verify the inputs to the reinforce_train function.
    """
    for context, scores, mask in zip(contexts_list, scores_list, output_masks_list):
        assert context.size(0) == scores.size(0) == mask.size(0), (
            f"Context, scores, and mask lengths do not match. "
            f"Context shape: {context.shape}, scores shape: {scores.shape}, Mask shape: {mask.shape}"
        )


def output_train_data_debug(
    path, contexts_list, scores_list, output_masks_list, tokenizer
):
    """
    Output the training data for debugging.

    Args:
        path (str): The directory path where the output files will be saved.
        contexts_list (list of torch.Tensor): List of input contexts, each of shape (S, V).
        scores_list (list of torch.Tensor): List of estimated scores for each time step, each of shape (S,).
        output_masks_list (list of torch.Tensor): List of masks for output tokens, each of shape (S,).
        tokenizer: Tokenizer to convert token IDs to their written form.
    """
    path = os.path.join(path, "train_debug", str(random.randint(0, 1000)))
    # Ensure the output directory exists
    os.makedirs(path, exist_ok=True)

    for idx, (context, scores, mask) in enumerate(
        zip(contexts_list, scores_list, output_masks_list)
    ):
        # Convert token IDs to written form
        tokens = tokenizer.convert_ids_to_tokens(context.tolist())

        # Prepare the triplets
        triplets = list(zip(tokens, scores.tolist(), mask.tolist()))

        # Define the file path for the current conversation
        file_path = os.path.join(path, f"conversation_{idx}.txt")

        # Write the triplets to the file
        with open(file_path, "w") as f:
            for token, ret, msk in triplets:
                f.write(f"{token}\t{ret}\t{msk}\n")
