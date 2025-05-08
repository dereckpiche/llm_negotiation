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

train_logger = logging.getLogger("train_logger")


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
    temperature=1.0,
    device="cuda:0",
    gradient_clipping=None,
    debug_log_path=None,
    debug_enabled=False,
):
    """Trains a language model using the REINFORCE algorithm (policy gradient method).

    This function implements policy gradient reinforcement learning, where the model is trained to maximize
    expected rewards by adjusting token probabilities based on reward signals. The training uses a batch
    of trajectories (context sequences, reward scores, and output masks) to update the model parameters.

    Args:
        model (torch.nn.Module): The language model with a causal LM head to be optimized.
            This is typically a transformer model like GPT, LLaMA, etc.

        contexts_list (list of torch.Tensor): List of token ID sequences representing the input contexts.
            Each tensor has shape (S,) where S is the sequence length and contains integer token IDs.
            These represent the full conversation/generation history that we want to train on.
            During training, each token in position i will be used to predict the next token i+1,
            which is why we use contexts[:-1] and actions[1:] pairs in the code.
            Example: For the sentence "I like cats", the context would be the token IDs for
            ["I", "like"] when trying to predict "cats".

        scores_list (list of torch.Tensor): List of reward scores for each token position in the sequences.
            Each tensor has shape (S,) where S is the sequence length and contains float values.
            These scores represent the expected future reward at each token position and are used to
            weight the log probabilities in the policy gradient formula:
            - Positive scores encourage the model to increase the probability of those tokens
            - Negative scores discourage the model from generating those tokens
            - Zero scores have no effect on training
            These scores typically come from reward models, human feedback, or algorithmic rewards
            like RLOO (Return-Length-Optimized Oracle).

        output_masks_list (list of torch.Tensor): List of binary masks for valid output tokens.
            Each tensor has shape (S,) where S is the sequence length and contains binary values (0 or 1).
            The masks serve several crucial purposes:
            1. They specify which token positions should be trained on (1 = train, 0 = ignore)
            2. They handle variable-length sequences by masking padding tokens
            3. They can be used to focus training on specific parts of a sequence
            4. They prevent division by zero when normalizing gradients
            For example, in a dialogue we might only want to train on the assistant's responses,
            not the user inputs, so we would mask out the user's tokens.

        optimizer (torch.optim.Optimizer): Optimizer for training the model's parameters.
            Common choices include Adam, AdamW, or SGD.

        nb_epochs (int): Number of training epochs over the dataset.
            Each epoch processes the entire dataset once.

        mb_size (int): Minibatch size - the number of sequences processed together.
            Larger minibatches can improve training stability but require more memory.

        mb_per_step (int): Number of minibatches to accumulate gradients over before taking an optimizer step.
            If set to -1, will compute based on dataset size. This enables effective larger batch sizes
            without the memory requirements, by accumulating gradients across multiple forward/backward passes.

        output_path (str, optional): If provided, debug information about the training data will be saved here.

        tokenizer: Tokenizer to convert token IDs to their text form for debugging purposes.

        gradient_checkpointing (bool): Whether to use gradient checkpointing to save memory.
            This trades computational efficiency for reduced memory usage.

        entropy_coef (float): Coefficient for entropy bonus in the loss function.
            Higher values encourage more random/diverse outputs.

        kl_loss_coef (float): Coefficient for KL divergence penalty to prevent large policy shifts.
            Higher values make training more conservative.

        temperature (float): Controls the sharpness of the softmax distribution during training.
            Values > 1 make the distribution more uniform, < 1 make it more peaked.
            Must be greater than 0.

        device (str): The device to run training on, e.g., "cuda:0", "cpu".

        gradient_clipping (float, optional): Maximum gradient norm for gradient clipping.
            Helps prevent exploding gradients.

        debug_log_path (str, optional): Path to the debug folder where training logs will be saved.
            A timestamped subfolder will be created for each reinforce_train call.

        debug_enabled (bool): Whether to enable detailed debugging output.

    Returns:
        dict: Dictionary containing training metrics, such as average loss.
    """
    train_logger.info(
        f"Memory taken before training: {torch.cuda.max_memory_allocated(device) / (1024 ** 2):.2f} MB"
    )
    train_logger.info(f"Starting reinforce training on {device}.")
    try:
        train_logger.info(f"Model active adapters are: {model.active_adapters}.")
    except:
        pass
    try:
        train_logger.info(f"Base model has data type {model.base_model_torch_dtype}.")
    except:
        pass
    try:
        train_logger.info(
            f"Model has {model.get_nb_trainable_parameters()} trainable parameters."
        )
    except:
        pass
    try:
        train_logger.info(f"Model has PEFT type {model.peft_type}.")
    except:
        pass
    try:
        train_logger.info(
            f"Model base class is {model.get_nb_trainable_parameters()} trainable parameters."
        )
    except:
        pass
    model.train()
    if gradient_checkpointing == True:
        train_logger.info("Enabling gradient checkpointing")
        model.gradient_checkpointing_enable(dict(use_reentrant=False))

    if optimizer is None:
        raise ValueError(
            "Optimizer must be provided. Please pass an optimizer instance."
        )

    verify_reinforce_train_inputs(contexts_list, scores_list, output_masks_list)

    # Calculate the maximum context length in terms of number of tokens
    max_context_length = max(context.size(0) for context in contexts_list)

    max_memory_usage = 0  # Initialize max memory usage

    # nb_trajectories_we_train_on is now the multiple of mb_size
    nb_trajectories_we_train_on = len(contexts_list) - len(contexts_list) % mb_size
    if mb_per_step == -1:
        gradient_accumulation_steps = nb_trajectories_we_train_on // mb_size
    else:
        gradient_accumulation_steps = mb_per_step
    train_logger.info(f"Gradient accumulation steps are {gradient_accumulation_steps}.")

    # Initialize Accelerator
    model_accelerator = Accelerator()
    model, optimizer = model_accelerator.prepare(model, optimizer)

    # Configure debug logging with timestamped subfolder
    if debug_enabled and debug_log_path is not None:
        import datetime
        import os

        # Create a timestamped subfolder for this training run
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_subfolder = os.path.join(
            debug_log_path, f"reinforce_training_debug_logs_{timestamp}"
        )
        os.makedirs(debug_subfolder, exist_ok=True)

        # Store initial trainable weights for comparison
        _log_trainable_weights_snapshot(
            model, os.path.join(debug_subfolder, "trainable_weights_initial.json")
        )

        debug_log_path = debug_subfolder

    # Determine if debug logging is enabled
    debug_enabled = debug_enabled and debug_log_path is not None
    if debug_enabled and tokenizer is None:
        # Try to find a tokenizer from the model
        tokenizer = find_tokenizer_from_model(model)
        debug_enabled = debug_enabled and tokenizer is not None

    train_output_dict = defaultdict(list)

    for epoch in range(nb_epochs):
        train_logger.info(f"Epoch {epoch} of {nb_epochs}")
        for i in range(0, len(contexts_list), mb_size):
            train_logger.info(f"Batch {i} of {len(contexts_list)}")
            # Get the minibatch
            context_batch = contexts_list[i : i + mb_size]
            return_batch = scores_list[i : i + mb_size]
            mask_batch = output_masks_list[i : i + mb_size]

            action_batch = [a[1:] for a in context_batch]
            return_batch = [r[1:] for r in return_batch]
            mask_batch = [m[1:] for m in mask_batch]
            context_batch = [c[:-1] for c in context_batch]

            # Track if this is the first batch of training for verification
            is_first_batch = epoch == 0 and i == 0

            # Pad sequences
            action_batch = pad_sequence(action_batch, batch_first=True).long()
            context_batch = pad_sequence(context_batch, batch_first=True).long()
            return_batch = pad_sequence(return_batch, batch_first=True).float()
            mask_batch = pad_sequence(mask_batch, batch_first=True).float()

            # Create attention mask to ignore padding tokens
            attention_mask = (
                context_batch != 0
            ).long()  # context_batch: (B, S) -> attention_mask: (B, S)

            device = model_accelerator.device
            # Move data to the appropriate device
            action_batch = action_batch.to(device)  # (B, S)
            context_batch = context_batch.to(device)  # (B, S)
            return_batch = return_batch.to(device)  # (B, S)
            mask_batch = mask_batch.to(device)  # (B, S)
            attention_mask = attention_mask.to(device)  # (B, S)

            # Forward pass
            logits = model(input_ids=context_batch, attention_mask=attention_mask)[
                0
            ]  # (B, S, V)

            # Apply temperature scaling before computing log probabilities
            assert temperature > 0, "Temperature must be greater than 0."

            logits /= temperature  # (B, S, V)

            # Compute new log probabilities
            log_probs = F.log_softmax(logits, dim=-1)  # (B, S, V)

            # Apply mask to log probabilities and values
            action_log_probs = log_probs.gather(
                dim=-1, index=action_batch.unsqueeze(-1)
            ).squeeze(
                -1
            )  # (B,S)

            rewarded_action_log_probs = action_log_probs * (
                return_batch * mask_batch
            )  # (B, S)

            # Compute entropy
            if entropy_coef != 0.0:
                # Dimensions after Softmax are (B, S, V). Chaining operations to reduce memory overhead.
                entropy = (-log_probs * F.softmax(logits, dim=-1)).sum(dim=-1)  # (B, S)

                rewarded_action_log_probs += entropy_coef * (
                    entropy * mask_batch
                )  # (B, S)

                del entropy

            del log_probs

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

                del kl_div

                rewarded_action_log_probs -= per_token_kl

                del per_token_kl

            # Avoid division by zero by adding a small epsilon value to the denominator
            epsilon = 1e-8

            loss = -rewarded_action_log_probs.sum(dim=1) / (
                torch.sum(mask_batch, dim=1) + epsilon
            )  # (B,)

            loss = loss.mean()

            train_output_dict["loss"].append(loss.item())

            # ------------------------------------------------------------------
            # Per‑batch debug logging (after computing loss)
            # ------------------------------------------------------------------
            if debug_enabled:
                _output_reinforce_step_debug(
                    save_dir=debug_log_path,
                    tokenizer=tokenizer,
                    epoch=epoch,
                    batch_start_idx=i,
                    context_batch=context_batch,
                    action_batch=action_batch,
                    logits=logits,
                    action_log_probs=action_log_probs,
                    return_batch=return_batch,
                    mask_batch=mask_batch,
                    entropy=0,
                    per_token_kl=0,
                )

            loss /= gradient_accumulation_steps
            model_accelerator.backward(loss)
            train_logger.info(f"Accumulated a gradient on loss of {loss.item()}.")

            del action_batch
            del context_batch
            del return_batch
            del mask_batch
            del attention_mask

            del logits
            del loss
            del action_log_probs
            del rewarded_action_log_probs

            torch.cuda.empty_cache()

            if ((i + mb_size) // mb_size) % gradient_accumulation_steps == 0:
                if gradient_clipping and model_accelerator.sync_gradients:
                    # norm is computed by concatenating all gradients, the gradients are present for param with requires_grad
                    grad_norm = model_accelerator.clip_grad_norm_(
                        model.parameters(), gradient_clipping
                    )
                    train_output_dict["grad_norm"].append(grad_norm.item())

                if debug_enabled and debug_log_path is not None:
                    grad_snapshot_path = os.path.join(
                        debug_log_path,
                        f"gradient_snapshot_epoch{epoch}_batch{i}.json",
                    )
                    _log_trainable_weights_gradient_snapshot(model, grad_snapshot_path)

                train_logger.info(f"Applying a gradient step on the model.")
                optimizer.step()
                optimizer.zero_grad()

            # Update max GPU memory usage
            current_memory_usage = torch.cuda.max_memory_allocated(device)
            if current_memory_usage > max_memory_usage:
                max_memory_usage = current_memory_usage

    train_logger.info(
        f"Max GPU memory usage during entire training: {max_memory_usage / (1024 ** 2):.2f} MB"
    )

    # Save final trainable weights snapshot if debug enabled
    if debug_enabled:
        _log_trainable_weights_snapshot(
            model, os.path.join(debug_log_path, "trainable_weights_final.json")
        )

        # Generate trainable weights comparison report
        _generate_trainable_weights_comparison(
            os.path.join(debug_log_path, "trainable_weights_initial.json"),
            os.path.join(debug_log_path, "trainable_weights_final.json"),
            os.path.join(debug_log_path, "trainable_weights_comparison.txt"),
        )

    model_accelerator.clear(model, optimizer)
    for key in train_output_dict:
        train_output_dict[key] = np.mean(train_output_dict[key])
    # TODO: log memory taken
    train_logger.info(
        f"Memory taken after training: {torch.cuda.max_memory_allocated(device) / (1024 ** 2):.2f} MB"
    )
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    return train_output_dict


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


# =====================================================================================================
# =====================================================================================================
# Debug helpers
# =====================================================================================================
# =====================================================================================================


def _log_trainable_weights_snapshot(model, save_path):
    """
    Creates a snapshot of all trainable weights for verification.

    Extracts and saves all parameters with gradients enabled to verify if they're being
    updated during training. This function captures all trainable parameters, categorizing
    them as either adapter parameters or base model parameters for analysis.

    Args:
        model: The model containing trainable parameters
        save_path: Path to save the trainable weights snapshot
    """
    import json
    import os
    from collections import defaultdict

    import numpy as np
    import torch

    # Dictionary to store trainable weights
    trainable_weights = defaultdict(dict)

    # Helper to determine if a module is likely an adapter
    def is_adapter_param(name):
        adapter_keywords = ["lora", "adapter", "peft", "prefix"]
        return any(keyword in name.lower() for keyword in adapter_keywords)

    # Helper to make tensors JSON serializable with high precision
    def tensor_to_serializable(tensor, max_values=10):
        if tensor is None:
            return None

        # Convert to list of floats (first few values) with high precision
        if tensor.numel() > 0:
            flat_tensor = tensor.detach().flatten()
            # Store with full float64 precision to catch even tiny changes
            first_values = flat_tensor[:max_values].to(torch.float64).tolist()
            last_values = (
                flat_tensor[-max_values:].to(torch.float64).tolist()
                if tensor.numel() > max_values
                else []
            )

            # Create a unique fingerprint for the tensor that works with any dtype
            # Convert to float32 first to handle BFloat16 and other special types
            try:
                tensor_hash = hash(
                    tensor.detach().cpu().to(torch.float32).numpy().tobytes()
                )
            except Exception:
                # Fallback to a simpler fingerprint if hashing fails
                tensor_hash = hash(str(first_values) + str(last_values))

            # Compute stats with high precision
            return {
                "shape": list(tensor.shape),
                "first_values": first_values,
                "last_values": last_values,
                "norm": float(tensor.norm().item()),
                "mean": float(tensor.mean().item()),
                "std": float(tensor.std().item()) if tensor.numel() > 1 else 0.0,
                "min": float(tensor.min().item()),
                "max": float(tensor.max().item()),
                "fingerprint": tensor_hash,
                "numel": tensor.numel(),
            }
        return {"empty": True, "shape": list(tensor.shape)}

    # Extract all trainable parameters
    trainable_params = {}
    frozen_params = {}

    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad:  # Capture all trainable parameters
                trainable_params[name] = tensor_to_serializable(param)
            else:
                # For frozen params, just store summary stats to save space
                try:
                    frozen_params[name] = {
                        "shape": list(param.shape),
                        "norm": float(param.norm().item()) if param.numel() > 0 else 0,
                        "requires_grad": param.requires_grad,
                    }
                except Exception as e:
                    # print(f"Error computing norm for {name}: {e}")
                    frozen_params[name] = {
                        "shape": list(param.shape),
                        "requires_grad": param.requires_grad,
                    }

    # Find PEFT modules more explicitly for categorization purposes
    peft_modules = []
    training_info = {"has_adapters": False, "adapter_types": []}

    for name, module in model.named_modules():
        # Check for PEFT attributes
        if (
            hasattr(module, "lora_A")
            or hasattr(module, "lora_B")
            or "lora" in name.lower()
            or "adapter" in name.lower()
        ):
            peft_modules.append(name)
            training_info["has_adapters"] = True
            if "lora" not in training_info["adapter_types"]:
                training_info["adapter_types"].append("lora")

    # Categorize trainable parameters
    adapter_params = {}
    base_model_params = {}

    for name, param_info in trainable_params.items():
        if is_adapter_param(name):
            adapter_params[name] = param_info
        else:
            base_model_params[name] = param_info

    # Store everything in a dictionary
    training_info.update(
        {
            "all_trainable_parameters": trainable_params,
            "adapter_parameters": adapter_params,
            "base_model_parameters": base_model_params,
            "frozen_parameters_summary": frozen_params,
            "peft_modules": peft_modules,
            "trainable_params_count": len(trainable_params),
            "adapter_params_count": len(adapter_params),
            "base_model_params_count": len(base_model_params),
            "frozen_params_count": len(frozen_params),
            "trainable_param_sizes": {
                name: info["shape"] for name, info in trainable_params.items()
            },
        }
    )

    # Save to file
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(training_info, f, indent=2)

    return training_info


def _generate_trainable_weights_comparison(initial_path, final_path, output_path):
    """
    Generates a human-readable comparison between initial and final trainable weights.
    Shows differences in scientific notation and marks changes with visual indicators.
    This works for all trainable parameters, including both adapter and base model weights.

    Args:
        initial_path: Path to the initial trainable weights JSON
        final_path: Path to the final trainable weights JSON
        output_path: Path to save the comparison report
    """
    import json
    import os
    from collections import defaultdict

    import numpy as np

    try:
        # Load initial and final weights
        with open(initial_path, "r") as f:
            initial = json.load(f)

        with open(final_path, "r") as f:
            final = json.load(f)

        # Prepare comparison report
        with open(output_path, "w") as f:
            f.write("TRAINABLE WEIGHTS COMPARISON REPORT\n")
            f.write("==================================\n\n")

            # High-level summary
            initial_params = initial.get("trainable_params_count", 0)
            final_params = final.get("trainable_params_count", 0)
            f.write(f"Initial trainable parameters: {initial_params}\n")
            f.write(f"Final trainable parameters: {final_params}\n\n")

            # Check if there are adapter parameters
            has_adapters = initial.get("has_adapters", False) or final.get(
                "has_adapters", False
            )
            if has_adapters:
                initial_adapter_params = initial.get("adapter_params_count", 0)
                final_adapter_params = final.get("adapter_params_count", 0)
                initial_base_params = initial.get("base_model_params_count", 0)
                final_base_params = final.get("base_model_params_count", 0)

                f.write(f"Initial adapter parameters: {initial_adapter_params}\n")
                f.write(f"Final adapter parameters: {final_adapter_params}\n")
                f.write(f"Initial base model parameters: {initial_base_params}\n")
                f.write(f"Final base model parameters: {final_base_params}\n\n")

            # Compare trainable parameters
            init_params = initial.get("all_trainable_parameters", {})
            final_params = final.get("all_trainable_parameters", {})

            # Use sorted keys from initial parameters to ensure consistent ordering
            all_param_names = sorted(
                set(list(init_params.keys()) + list(final_params.keys()))
            )

            f.write(f"Detailed parameter comparison:\n")
            f.write(f"----------------------------\n\n")

            # Track overall change statistics
            total_changed = 0
            total_identical = 0
            total_params = 0
            adapter_changed = 0
            base_model_changed = 0

            # Helper to identify adapter parameters
            def is_adapter_param(name):
                adapter_keywords = ["lora", "adapter", "peft", "prefix"]
                return any(keyword in name.lower() for keyword in adapter_keywords)

            for name in all_param_names:
                total_params += 1
                is_adapter = is_adapter_param(name)
                param_type = "ADAPTER" if is_adapter else "BASE_MODEL"

                if name in init_params and name in final_params:
                    init_tensor = init_params[name]
                    final_tensor = final_params[name]

                    # Check for exact equality using fingerprint if available
                    exact_match = init_tensor.get(
                        "fingerprint", None
                    ) == final_tensor.get("fingerprint", None)

                    # Calculate differences for various metrics
                    changes = {}

                    # Norm difference
                    init_norm = init_tensor.get("norm", 0)
                    final_norm = final_tensor.get("norm", 0)
                    norm_diff = final_norm - init_norm

                    # Check for any changes in values (even extremely small ones)
                    init_first = init_tensor.get("first_values", [])
                    final_first = final_tensor.get("first_values", [])

                    any_changes = not exact_match
                    value_diffs = []

                    for i, (iv, fv) in enumerate(zip(init_first, final_first)):
                        diff = fv - iv
                        if abs(diff) > 0:
                            any_changes = True
                        value_diffs.append(diff)

                    # Set status indicator
                    if any_changes:
                        status = "✓"  # Changed
                        total_changed += 1
                        if is_adapter:
                            adapter_changed += 1
                        else:
                            base_model_changed += 1
                    else:
                        status = "✗"  # Identical
                        total_identical += 1

                    # Write parameter comparison
                    f.write(f"{status} Parameter: {name} [{param_type}]\n")
                    f.write(f"  - Shape: {init_tensor.get('shape')}\n")
                    f.write(
                        f"  - Norm: {init_norm:.8e} → {final_norm:.8e} (diff: {norm_diff:.8e})\n"
                    )

                    # Add other statistics in scientific notation
                    for stat in ["mean", "std", "min", "max"]:
                        if stat in init_tensor and stat in final_tensor:
                            init_val = init_tensor[stat]
                            final_val = final_tensor[stat]
                            diff = final_val - init_val
                            f.write(
                                f"  - {stat.capitalize()}: {init_val:.8e} → {final_val:.8e} (diff: {diff:.8e})\n"
                            )

                    # Show detailed value comparisons
                    if init_first and final_first:
                        f.write("  - Values comparison:\n")
                        for i, (iv, fv) in enumerate(
                            zip(init_first[:5], final_first[:5])
                        ):
                            diff = fv - iv
                            change_indicator = "✓" if abs(diff) > 0 else "✗"
                            f.write(
                                f"      [{i}] {change_indicator} {iv:.16e} → {fv:.16e} (diff: {diff:.16e})\n"
                            )

                    f.write("\n")
                else:
                    # Parameter exists in only one snapshot
                    f.write(
                        f"! Parameter: {name} [{param_type}] - exists in {'initial' if name in init_params else 'final'} snapshot only\n\n"
                    )

            # Overall change summary
            f.write("\nChange Summary:\n")
            f.write("==============\n")
            f.write(f"Total parameters: {total_params}\n")
            f.write(
                f"Parameters with ANY change: {total_changed} ({(total_changed/total_params)*100:.2f}%)\n"
            )
            f.write(
                f"Parameters with NO change: {total_identical} ({(total_identical/total_params)*100:.2f}%)\n"
            )

            if has_adapters:
                adapter_total = sum(
                    1 for name in all_param_names if is_adapter_param(name)
                )
                base_model_total = total_params - adapter_total

                if adapter_total > 0:
                    f.write(
                        f"Adapter parameters changed: {adapter_changed}/{adapter_total} ({(adapter_changed/adapter_total)*100:.2f}%)\n"
                    )

                if base_model_total > 0:
                    f.write(
                        f"Base model parameters changed: {base_model_changed}/{base_model_total} ({(base_model_changed/base_model_total)*100:.2f}%)\n"
                    )

            # Conclusion
            if total_changed > 0:
                f.write("\nCONCLUSION: Trainable weights changed during training.\n")
            else:
                f.write(
                    "\nCONCLUSION: NO CHANGES detected in trainable weights! Training had no effect.\n"
                )

    except Exception as e:
        pass


def verify_reinforce_train_inputs(contexts_list, scores_list, output_masks_list):
    """
    Verify the inputs to the reinforce_train function.
    """
    for context, scores, mask in zip(contexts_list, scores_list, output_masks_list):
        assert context.size(0) == scores.size(0) == mask.size(0), (
            f"Context, scores, and mask lengths do not match. "
            f"Context shape: {context.shape}, scores shape: {scores.shape}, Mask shape: {mask.shape}"
        )


def _log_trainable_weights_gradient_snapshot(model, save_path):
    """
    Creates a snapshot of gradients for all trainable weights.

    This function captures gradient information for parameters with gradients enabled
    right before optimizer.step() to verify if meaningful gradients are being computed.

    Args:
        model: The model containing trainable parameters
        save_path: Path to save the gradient snapshot
    """
    import json
    import os
    from collections import defaultdict

    import numpy as np
    import torch

    # Dictionary to store gradient information
    gradient_info = defaultdict(dict)

    # Helper to determine if a module is likely an adapter
    def is_adapter_param(name):
        adapter_keywords = ["lora", "adapter", "peft", "prefix"]
        return any(keyword in name.lower() for keyword in adapter_keywords)

    # Helper to make tensors JSON serializable with high precision
    def tensor_to_serializable(tensor, max_values=10):
        if tensor is None:
            return None

        # Convert to list of floats (first few values) with high precision
        if tensor.numel() > 0:
            flat_tensor = tensor.detach().flatten()
            # Store with full float64 precision to catch even tiny gradients
            first_values = flat_tensor[:max_values].to(torch.float64).tolist()

            # Create stats for the tensor
            return {
                "shape": list(tensor.shape),
                "first_values": first_values,
                "norm": float(tensor.norm().item()),
                "mean": float(tensor.mean().item()),
                "std": float(tensor.std().item()) if tensor.numel() > 1 else 0.0,
                "min": float(tensor.min().item()),
                "max": float(tensor.max().item()),
                "has_nan": bool(torch.isnan(tensor).any().item()),
                "has_inf": bool(torch.isinf(tensor).any().item()),
                "numel": tensor.numel(),
            }
        return {"empty": True, "shape": list(tensor.shape)}

    # Extract gradient information for all trainable parameters
    trainable_params_with_grad = {}
    trainable_params_without_grad = {}

    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Check if the parameter has a gradient
                if param.grad is not None:
                    trainable_params_with_grad[name] = {
                        "param_info": tensor_to_serializable(param),
                        "grad_info": tensor_to_serializable(param.grad),
                        "is_adapter": is_adapter_param(name),
                    }
                else:
                    trainable_params_without_grad[name] = {
                        "param_info": tensor_to_serializable(param),
                        "is_adapter": is_adapter_param(name),
                    }

    # Compute overall statistics
    gradient_info["stats"] = {
        "total_trainable_params": len(trainable_params_with_grad)
        + len(trainable_params_without_grad),
        "params_with_grad": len(trainable_params_with_grad),
        "params_without_grad": len(trainable_params_without_grad),
        "adapter_params_with_grad": sum(
            1 for p in trainable_params_with_grad.values() if p["is_adapter"]
        ),
        "base_model_params_with_grad": sum(
            1 for p in trainable_params_with_grad.values() if not p["is_adapter"]
        ),
    }

    # Compute aggregate gradient norms by layer type
    layer_prefixes = defaultdict(list)
    for name, info in trainable_params_with_grad.items():
        # Extract layer prefix (e.g., 'model.layers.0', 'lm_head')
        parts = name.split(".")
        if len(parts) >= 2:
            prefix = ".".join(parts[:2])
        else:
            prefix = parts[0]

        if "grad_info" in info and "norm" in info["grad_info"]:
            layer_prefixes[prefix].append(info["grad_info"]["norm"])

    # Calculate average gradient norm by layer
    gradient_info["layer_grad_norms"] = {
        prefix: {
            "mean_grad_norm": np.mean(norms),
            "max_grad_norm": max(norms),
            "num_params": len(norms),
        }
        for prefix, norms in layer_prefixes.items()
    }

    # Store detailed gradient information
    gradient_info["params_with_grad"] = trainable_params_with_grad
    gradient_info["params_without_grad"] = trainable_params_without_grad

    # Save to file
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(gradient_info, f, indent=2)

    # Return overall stats for logging
    return gradient_info["stats"]


def _output_reinforce_step_debug(
    save_dir,
    tokenizer,
    epoch,
    batch_start_idx,
    context_batch,
    action_batch,
    logits,
    action_log_probs,
    return_batch,
    mask_batch,
    entropy,
    per_token_kl,
):
    """Save focused debug logs for validating reinforcement training.

    Efficiently logs information to verify we're reinforcing the right actions on the right contexts.
    Creates lightweight logs that focus on critical validation of the reinforcement process.

    Args:
        save_dir (str): root directory for debug logs.
        tokenizer: HuggingFace tokenizer used for decoding.
        epoch (int): current epoch index.
        batch_start_idx (int): index of first trajectory in current minibatch.
        context_batch (Tensor): (B, S) input ids (excluding last token).
        action_batch (Tensor): (B, S) target tokens (next token for each position).
        logits (Tensor): (B, S, V) raw logits from model (already temperature‑scaled).
        action_log_probs (Tensor): (B, S) log probs of chosen actions.
        return_batch (Tensor): (B, S) reward/score tensor.
        mask_batch (Tensor): (B, S) output mask tensor.
        entropy (Tensor): (B, S) entropy per token.
        per_token_kl (Tensor): (B, S) kl contribution per token (already scaled by kl_loss_coef).
    """
    import json
    import os
    from collections import defaultdict

    import numpy as np
    import torch

    # Create batch directory with a more efficient structure
    epoch_dir = os.path.join(save_dir, f"epoch_{epoch:03d}")
    batch_idx = batch_start_idx // context_batch.size(0) if context_batch.size(0) else 0
    batch_dir = os.path.join(epoch_dir, f"batch_{batch_idx:06d}")
    os.makedirs(batch_dir, exist_ok=True)

    # Keep tensors on their original device and compute metrics there
    B, S = context_batch.shape
    vocab_dim = logits.shape[-1]

    # Efficiently compute batch-level metrics on GPU
    total_reward = (return_batch * mask_batch).sum().item()
    total_tokens = mask_batch.sum().item()
    avg_reward_per_token = total_reward / total_tokens if total_tokens > 0 else 0

    # Calculate effective reinforcement (log_prob * reward) on device
    effective_reinforcement = action_log_probs * return_batch * mask_batch
    avg_reinforcement = (
        effective_reinforcement.sum().item() / total_tokens if total_tokens > 0 else 0
    )

    # Only analyze a sample of sequences for detailed validation (max 4)
    sample_indices = list(range(min(4, B)))

    # Process sample sequences for verification
    for sample_idx, s in enumerate(sample_indices):
        # Get tensors for this sequence - keep on device for computation
        input_ids = context_batch[s]
        target_ids = action_batch[s]
        sample_mask = mask_batch[s]
        sample_returns = return_batch[s]
        sample_log_probs = action_log_probs[s]
        sample_effective_reinforcement = effective_reinforcement[s]

        # Count valid tokens (where mask > 0)
        valid_tokens = int(sample_mask.sum().item())

        # Only process up to the actual sequence length, not padding
        seq_len = (input_ids != 0).sum().item()

        # First, decode the entire sequence to show the full context
        # Convert all non-zero tokens to a list
        valid_input_ids = [id.item() for id in input_ids if id != 0]
        full_sequence = tokenizer.decode(valid_input_ids, skip_special_tokens=False)

        # Create verification records - focus on ACTION-CONTEXT pairs
        verification_records = []

        # Only examine masked tokens for verification to speed things up
        valid_indices = torch.where(sample_mask > 0)[0]

        # Process all valid indices, no limit
        for t_idx in valid_indices:
            t = t_idx.item()

            # Skip if we're out of sequence bounds
            if t >= seq_len:
                continue

            # Get the full context up to this point - this is the actual context length used
            # for prediction at this position
            full_context_ids = input_ids[: t + 1].tolist()
            # Filter out padding tokens (0s)
            full_context_ids = [id for id in full_context_ids if id != 0]
            actual_context_length = len(full_context_ids)

            # Get a window of context for display (up to 20 tokens to keep it readable)
            context_window_size = min(5, actual_context_length)
            start_idx = max(0, t + 1 - context_window_size)
            context_window_ids = input_ids[start_idx : t + 1].tolist()
            context_window_ids = [id for id in context_window_ids if id != 0]

            # Create readable context with special characters for spaces
            # Decode the tokens individually to preserve exact token boundaries
            context_tokens = []
            for ctx_id in context_window_ids:
                token_text = tokenizer.decode([ctx_id])
                # Replace spaces with visible space character and newlines with \n
                token_text = token_text.replace(" ", "·").replace("\n", "↵")
                context_tokens.append(token_text)

            # Include the full context window and current token
            context_str = "".join(context_tokens)
            current_token = context_tokens[-1] if context_tokens else "[START]"

            # Also get the full raw context by decoding all context window tokens at once
            # This gives us the original text without token boundaries
            full_context = tokenizer.decode(context_window_ids)
            full_context = full_context.replace(" ", "·").replace("\n", "↵")

            # Target token - format it with visible spaces
            target_token_id = target_ids[t].item()
            target_token_text = tokenizer.decode([target_token_id])
            target_token_text = target_token_text.replace(" ", "·").replace("\n", "↵")

            # Verify if this token will be correctly reinforced
            is_masked = bool(sample_mask[t].item() > 0)
            has_reward = abs(sample_returns[t].item()) > 1e-6
            has_log_prob = abs(sample_log_probs[t].item()) > 1e-6

            will_be_reinforced = is_masked and has_reward and has_log_prob
            reinforcement_value = (
                sample_effective_reinforcement[t].item() if will_be_reinforced else 0
            )

            # Create a record focused on action-context verification
            record = {
                "position": t,
                "context_window": context_str,
                "full_context": full_context,
                "actual_context_length": actual_context_length,
                "current_token": current_token,
                "target_token": target_token_text,
                "reward": float(sample_returns[t].item()),
                "log_prob": float(sample_log_probs[t].item()),
                "effective_reinforcement": reinforcement_value,
                "verification": {
                    "is_masked": is_masked,
                    "has_reward": has_reward,
                    "has_log_prob": has_log_prob,
                    "will_be_reinforced": will_be_reinforced,
                },
            }
            verification_records.append(record)

        # Save a text-based summary for quick verification
        txt_path = os.path.join(batch_dir, f"verification_{sample_idx}.txt")
        with open(txt_path, "w") as f:
            f.write(f"SEQUENCE {sample_idx} VERIFICATION\n")
            f.write(f"Valid tokens: {valid_tokens}/{seq_len}\n")
            f.write(
                f"Average reward: {sample_returns[sample_mask > 0].mean().item():.4f}\n\n"
            )

            # Show the full sequence first for complete context
            f.write("FULL SEQUENCE:\n")
            f.write("-" * 100 + "\n")
            # Replace spaces with visible markers to make them obvious
            # Replace newlines with ↵ symbol but don't create actual line breaks
            visible_sequence = full_sequence.replace(" ", "·").replace("\n", "↵")
            f.write(visible_sequence)
            f.write("\n\n")

            f.write("CONTEXT → TARGET TOKEN REINFORCEMENT:\n")
            f.write("-" * 100 + "\n")

            # Define thresholds for significant reinforcement
            reinforcement_threshold = (
                1.0  # Adjust based on your typical reinforcement values
            )

            for rec in verification_records:
                # Determine reinforcement indicator based on effective reinforcement
                # In REINFORCE, the update direction depends on the reward sign, not the effective_reinforcement directly
                ef_effective_reinforcement = rec["effective_reinforcement"]
                reward = rec["reward"]

                if abs(ef_effective_reinforcement) < reinforcement_threshold:
                    reinforcement_marker = "━"  # Negligible reinforcement
                elif reward > 0:
                    # Positive reward means we want to INCREASE the probability of this action
                    # (even though effective_reinforcement might be negative due to negative log_prob)
                    reinforcement_marker = (
                        "✓"  # Positive reinforcement (increase probability)
                    )
                else:
                    # Negative reward means we want to DECREASE the probability of this action
                    reinforcement_marker = (
                        "✗"  # Negative reinforcement (decrease probability)
                    )

                # Get last 10 characters of context window instead of full token list
                context_text = rec["full_context"]
                last_10_chars = context_text[-10:].ljust(10)

                # Calculate regular probability from log probability
                log_prob = rec["log_prob"]
                prob = np.exp(log_prob)

                # Format the line with consistent spacing
                context_info = f"[{rec['actual_context_length']}] {last_10_chars}"

                # Format the line with context window, ensuring sufficient width
                f.write(
                    f"{rec['position']:3d} | {context_info:40s} → {rec['target_token']:15s} | "
                )
                f.write(
                    f"r={rec['reward']:+.2f} | logp={log_prob:.2f} | p={prob:.3f} | eff={ef_effective_reinforcement:+.2f} | {reinforcement_marker}\n"
                )

        # Save compact JSON for programmatic analysis
        json_data = {"full_sequence": full_sequence, "records": verification_records}
        json_data = _make_json_serializable(json_data)

        json_path = os.path.join(batch_dir, f"verification_{sample_idx}.json")
        with open(json_path, "w") as f:
            json.dump(json_data, f)

    # Return lightweight summary
    return {
        "batch_idx": batch_idx,
        "total_reward": total_reward,
        "avg_reinforcement": avg_reinforcement,
        "valid_token_percentage": float(total_tokens / (B * S)) if B * S > 0 else 0,
    }


def _log_model_adapter_status(model, save_dir, epoch, batch_idx):
    """
    Log information about which adapters are active in the model
    and their parameter counts.

    Args:
        model: The model being trained
        save_dir: Directory to save logs
        epoch: Current epoch
        batch_idx: Current batch index
    """
    import json
    import os
    from collections import defaultdict

    batch_dir = os.path.join(save_dir, f"epoch_{epoch:03d}", f"batch_{batch_idx:06d}")
    os.makedirs(batch_dir, exist_ok=True)

    # Detection of PEFT/LoRA adapters
    adapter_info = {
        "has_adapters": False,
        "adapter_types": [],
        "active_adapters": [],
        "total_trainable_params": 0,
        "total_frozen_params": 0,
        "total_params": 0,
        "trainable_percentage": 0.0,
        "modules_with_adapters": [],
    }

    def count_params(model):
        trainable, frozen = 0, 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable += param.numel()
            else:
                frozen += param.numel()
        return trainable, frozen

    # Check for PEFT attributes
    if hasattr(model, "peft_config"):
        adapter_info["has_adapters"] = True
        if isinstance(model.peft_config, dict):
            adapter_info["adapter_types"] = list(
                set(cfg.peft_type for cfg in model.peft_config.values())
            )
            adapter_info["active_adapters"] = list(model.peft_config.keys())
        else:
            adapter_info["adapter_types"] = [model.peft_config.peft_type]

    # Check for LoRA modules
    lora_modules = []
    for name, module in model.named_modules():
        if (
            "lora" in name.lower()
            or hasattr(module, "lora_A")
            or hasattr(module, "lora_B")
        ):
            lora_modules.append(name)
            adapter_info["has_adapters"] = True
            if "lora" not in adapter_info["adapter_types"]:
                adapter_info["adapter_types"].append("lora")

    adapter_info["modules_with_adapters"] = lora_modules

    # Check for active/disabled adapters
    if hasattr(model, "disable_adapter") or hasattr(model, "active_adapter"):
        adapter_info["has_adapters"] = True
        if hasattr(model, "active_adapter"):
            # Check if active_adapter is a method or an attribute
            if not callable(model.active_adapter):
                adapter_info["active_adapters"].append(model.active_adapter)
            else:
                # It's a method, so we can't serialize it directly
                adapter_info["active_adapters"].append("active_adapter_method_present")

    # Count parameters
    trainable_params, frozen_params = count_params(model)
    adapter_info["total_trainable_params"] = trainable_params
    adapter_info["total_frozen_params"] = frozen_params
    adapter_info["total_params"] = trainable_params + frozen_params
    if adapter_info["total_params"] > 0:
        adapter_info["trainable_percentage"] = (
            trainable_params / adapter_info["total_params"]
        ) * 100

    # Ensure all values are JSON serializable
    for key in list(adapter_info.keys()):
        if callable(adapter_info[key]):
            adapter_info[key] = str(adapter_info[key])

    # Make the entire structure JSON-serializable
    adapter_info = _make_json_serializable(adapter_info)

    # Save to file
    with open(os.path.join(batch_dir, "adapter_info.json"), "w") as f:
        json.dump(adapter_info, f, indent=2)

    return adapter_info


def _log_gradient_stats(
    model,
    optimizer,
    save_dir,
    epoch,
    batch_idx,
    gradient_accumulation_step,
    total_steps,
):
    """
    Log detailed statistics about gradients, including magnitude and accumulation status.

    Args:
        model: The model being trained
        optimizer: The optimizer
        save_dir: Directory to save logs
        epoch: Current epoch
        batch_idx: Current batch index
        gradient_accumulation_step: Current step in gradient accumulation
        total_steps: Total gradient accumulation steps
    """
    import json
    import os

    import numpy as np
    import torch

    batch_dir = os.path.join(save_dir, f"epoch_{epoch:03d}", f"batch_{batch_idx:06d}")
    os.makedirs(batch_dir, exist_ok=True)

    gradient_stats = {
        "gradient_accumulation": {
            "current_step": gradient_accumulation_step,
            "total_steps": total_steps,
            "is_accumulation_complete": gradient_accumulation_step == total_steps - 1,
        },
        "per_parameter_group": [],
        "overall": {
            "max_grad_norm": 0.0,
            "mean_grad_norm": 0.0,
            "median_grad_norm": 0.0,
            "num_params_with_grad": 0,
            "num_params_with_zero_grad": 0,
            "num_trainable_params": 0,
        },
    }

    # Collect all gradients for analysis
    all_grads = []
    all_grad_norms = []
    zero_grad_params = 0
    with_grad_params = 0
    trainable_params = 0

    # Group by parameter groups in optimizer
    for group_idx, group in enumerate(optimizer.param_groups):
        group_stats = {
            "group_idx": group_idx,
            "lr": group["lr"],
            "num_params": len(group["params"]),
            "weight_decay": group.get("weight_decay", 0),
            "momentum": group.get("momentum", 0),
            "beta1": group.get("betas", (0, 0))[0] if "betas" in group else None,
            "beta2": group.get("betas", (0, 0))[1] if "betas" in group else None,
            "grad_stats": {
                "max_norm": 0.0,
                "mean_norm": 0.0,
                "median_norm": 0.0,
                "num_zero_grad": 0,
            },
        }

        group_grads = []
        for p in group["params"]:
            if p.requires_grad:
                trainable_params += 1
                if p.grad is not None:
                    with_grad_params += 1
                    grad_norm = p.grad.norm().item()
                    all_grads.append(p.grad)
                    all_grad_norms.append(grad_norm)
                    group_grads.append(grad_norm)

                    # Check if gradient is actually zero
                    if grad_norm < 1e-10:
                        zero_grad_params += 1
                        group_stats["grad_stats"]["num_zero_grad"] += 1

        # Compute group statistics
        if group_grads:
            group_stats["grad_stats"]["max_norm"] = max(group_grads)
            group_stats["grad_stats"]["mean_norm"] = sum(group_grads) / len(group_grads)
            group_stats["grad_stats"]["median_norm"] = sorted(group_grads)[
                len(group_grads) // 2
            ]

        gradient_stats["per_parameter_group"].append(group_stats)

    # Overall statistics
    gradient_stats["overall"]["num_params_with_grad"] = with_grad_params
    gradient_stats["overall"]["num_params_with_zero_grad"] = zero_grad_params
    gradient_stats["overall"]["num_trainable_params"] = trainable_params

    if all_grad_norms:
        gradient_stats["overall"]["max_grad_norm"] = max(all_grad_norms)
        gradient_stats["overall"]["mean_grad_norm"] = sum(all_grad_norms) / len(
            all_grad_norms
        )
        gradient_stats["overall"]["median_grad_norm"] = sorted(all_grad_norms)[
            len(all_grad_norms) // 2
        ]

    # Analyze optimizer state for momentum
    if hasattr(optimizer, "state"):
        optimizer_state_stats = _analyze_optimizer_state(optimizer)
        gradient_stats["optimizer_state"] = optimizer_state_stats

    # Make the entire structure JSON-serializable
    gradient_stats = _make_json_serializable(gradient_stats)

    # Save gradient stats to file
    with open(
        os.path.join(
            batch_dir, f"gradient_stats_step_{gradient_accumulation_step}.json"
        ),
        "w",
    ) as f:
        json.dump(gradient_stats, f, indent=2)

    # Log key stats
    grad_status = (
        "zero"
        if with_grad_params == 0
        else f"mean={gradient_stats['overall']['mean_grad_norm']:.4e}"
    )

    return gradient_stats


def _analyze_optimizer_state(optimizer):
    """
    Analyze the optimizer state to understand momentum and other stateful values.

    Args:
        optimizer: The optimizer

    Returns:
        dict: Statistics about the optimizer state
    """
    state_stats = {
        "has_momentum": False,
        "has_exp_avg": False,
        "has_exp_avg_sq": False,
        "momentum_stats": {
            "mean_momentum_value": 0.0,
            "max_momentum_value": 0.0,
        },
        "num_params_with_state": 0,
    }

    # Skip if optimizer has no state
    if not hasattr(optimizer, "state") or len(optimizer.state) == 0:
        return state_stats

    # Analyze state
    momentum_values = []
    exp_avg_values = []
    exp_avg_sq_values = []

    for param_id, param_state in optimizer.state.items():
        state_stats["num_params_with_state"] += 1

        # Check for momentum (SGD)
        if "momentum_buffer" in param_state:
            state_stats["has_momentum"] = True
            momentum_buffer = param_state["momentum_buffer"]
            momentum_values.append(momentum_buffer.abs().mean().item())

        # Check for exponential moving averages (Adam, AdamW)
        if "exp_avg" in param_state:
            state_stats["has_exp_avg"] = True
            exp_avg = param_state["exp_avg"]
            exp_avg_values.append(exp_avg.abs().mean().item())

        if "exp_avg_sq" in param_state:
            state_stats["has_exp_avg_sq"] = True
            exp_avg_sq = param_state["exp_avg_sq"]
            exp_avg_sq_values.append(exp_avg_sq.abs().mean().item())

    # Calculate statistics for momentum
    if momentum_values:
        state_stats["momentum_stats"]["mean_momentum_value"] = sum(
            momentum_values
        ) / len(momentum_values)
        state_stats["momentum_stats"]["max_momentum_value"] = max(momentum_values)

    # Calculate statistics for Adam/AdamW state
    if state_stats["has_exp_avg"]:
        state_stats["adam_stats"] = {
            "mean_exp_avg": sum(exp_avg_values) / len(exp_avg_values)
            if exp_avg_values
            else 0,
            "max_exp_avg": max(exp_avg_values) if exp_avg_values else 0,
            "mean_exp_avg_sq": sum(exp_avg_sq_values) / len(exp_avg_sq_values)
            if exp_avg_sq_values
            else 0,
            "max_exp_avg_sq": max(exp_avg_sq_values) if exp_avg_sq_values else 0,
        }

    # Make sure the returned stats are JSON-serializable
    state_stats = _make_json_serializable(state_stats)

    return state_stats


def _log_step_overview(model, optimizer, step_summary, save_dir, epoch, batch_idx):
    """
    Generate a comprehensive per-step overview including all debug information.

    Args:
        model: The model being trained
        optimizer: The optimizer
        step_summary: Summary information from the training step
        save_dir: Directory to save logs
        epoch: Current epoch number
        batch_idx: Current batch index
    """
    import json
    import os

    import torch

    batch_dir = os.path.join(save_dir, f"epoch_{epoch:03d}", f"batch_{batch_idx:06d}")
    os.makedirs(batch_dir, exist_ok=True)

    # Combine all debugging information into a single comprehensive overview
    overview = {
        "step_info": {
            "epoch": epoch,
            "batch_idx": batch_idx,
            "gradient_accumulation_step": step_summary.get(
                "gradient_accumulation_step", None
            ),
            "total_gradient_accumulation_steps": step_summary.get(
                "total_gradient_accumulation_steps", None
            ),
        },
        "training_stats": {
            "loss": step_summary.get("loss", None),
            "reward": step_summary.get("reward", None),
            "reinforcement": step_summary.get("reinforcement", None),
        },
        "memory_stats": {
            "current_allocated": torch.cuda.memory_allocated()
            if torch.cuda.is_available()
            else 0,
            "max_allocated": torch.cuda.max_memory_allocated()
            if torch.cuda.is_available()
            else 0,
            "current_reserved": torch.cuda.memory_reserved()
            if torch.cuda.is_available()
            else 0,
        },
    }

    # Add timestamp
    from datetime import datetime

    overview["timestamp"] = datetime.now().isoformat()

    # Ensure the data is JSON-serializable
    overview = _make_json_serializable(overview)

    # Create a concise summary file
    with open(os.path.join(batch_dir, "step_overview.json"), "w") as f:
        json.dump(overview, f, indent=2)

    return overview


def _make_json_serializable(obj):
    """
    Recursively convert a nested structure (dict, list, etc.) to ensure all values are JSON-serializable.
    Non-serializable objects are converted to their string representation.

    Args:
        obj: The object to make JSON-serializable

    Returns:
        A JSON-serializable version of the object
    """
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_make_json_serializable(item) for item in obj)
    elif isinstance(obj, (int, float, bool, str, type(None))):
        return obj
    else:
        # Convert anything else to string representation
        try:
            # Try to convert to a basic type first
            if hasattr(obj, "item") and callable(obj.item):
                # For torch/numpy types with .item() method
                return obj.item()
            else:
                return str(obj)
        except:
            return f"<non-serializable: {type(obj).__name__}>"


def _log_optimization_step(model, optimizer, save_dir, epoch, batch_idx, pre_step=True):
    """
    Track parameter changes before and after an optimization step.

    Args:
        model: The model being trained
        optimizer: The optimizer
        save_dir: Directory to save logs
        epoch: Current epoch
        batch_idx: Current batch index
        pre_step: If True, this is called before the optimizer step, otherwise after
    """
    import json
    import os
    from collections import defaultdict

    import numpy as np
    import torch

    batch_dir = os.path.join(save_dir, f"epoch_{epoch:03d}", f"batch_{batch_idx:06d}")
    os.makedirs(batch_dir, exist_ok=True)

    # Create a unique filename for pre and post optimization step
    step_type = "pre_step" if pre_step else "post_step"
    file_path = os.path.join(batch_dir, f"optimization_{step_type}.json")

    # Gather parameter statistics
    param_stats = defaultdict(list)
    param_names = []

    # Sample a subset of parameters to avoid excessive logging
    max_params_to_sample = 10  # Limit sampling to avoid performance impact
    sampled_params = []

    with torch.no_grad():
        # Group parameters by module
        for name, param in model.named_parameters():
            if param.requires_grad:
                module_name = name.split(".")[0]
                param_stats["module_names"].append(module_name)

                # Only track detailed stats for a sample of parameters
                if len(sampled_params) < max_params_to_sample:
                    param_stats["param_names"].append(name)
                    param_stats["param_norms"].append(param.norm().item())
                    sampled_params.append((name, param.detach().clone()))

    # Calculate distribution of parameters across modules
    module_counts = {}
    for module in param_stats["module_names"]:
        if module not in module_counts:
            module_counts[module] = 0
        module_counts[module] += 1

    # Get optimizer state info
    optimizer_info = {}
    if hasattr(optimizer, "param_groups"):
        optimizer_info["num_param_groups"] = len(optimizer.param_groups)
        optimizer_info["param_group_sizes"] = [
            len(g["params"]) for g in optimizer.param_groups
        ]
        optimizer_info["learning_rates"] = [
            g.get("lr", 0) for g in optimizer.param_groups
        ]

    # Combine all information
    step_info = {
        "step_type": step_type,
        "epoch": epoch,
        "batch_idx": batch_idx,
        "module_distribution": module_counts,
        "optimizer_info": optimizer_info,
        "parameter_sample": {
            "names": param_stats["param_names"],
            "norms": param_stats["param_norms"],
        },
    }

    # Add parameter change tracking if this is post-step
    if not pre_step and hasattr(model, "_debug_pre_step_params"):
        changes = []
        for name, current_param in sampled_params:
            for pre_name, pre_param in model._debug_pre_step_params:
                if name == pre_name:
                    param_change = (current_param - pre_param).norm().item()
                    relative_change = param_change / (pre_param.norm().item() + 1e-8)
                    changes.append(
                        {
                            "param_name": name,
                            "absolute_change": param_change,
                            "relative_change": relative_change,
                        }
                    )
                    break

        step_info["parameter_changes"] = changes

        # Compute summary stats on changes
        if changes:
            abs_changes = [c["absolute_change"] for c in changes]
            rel_changes = [c["relative_change"] for c in changes]

            step_info["change_summary"] = {
                "mean_absolute_change": sum(abs_changes) / len(abs_changes),
                "max_absolute_change": max(abs_changes),
                "mean_relative_change": sum(rel_changes) / len(rel_changes),
                "max_relative_change": max(rel_changes),
            }

        # Clean up to avoid memory leaks
        delattr(model, "_debug_pre_step_params")

    # Store parameters for post-step comparison
    if pre_step:
        model._debug_pre_step_params = sampled_params

    # Make the entire structure JSON-serializable
    step_info = _make_json_serializable(step_info)

    # Save to file
    with open(file_path, "w") as f:
        json.dump(step_info, f, indent=2)

    return step_info


def _track_training_verification(
    model,
    optimizer,
    save_dir,
    epoch,
    accumulation_step,
    total_accumulation_steps,
    is_first_batch=False,
):
    """
    Explicitly tracks and verifies if gradients are properly accumulating and if the base model is being updated.

    This function provides clear yes/no confirmation about:
    1. Whether gradients are accumulating properly across steps
    2. Whether base model parameters (not just adapters) are being modified

    Args:
        model: The model being trained
        optimizer: The optimizer
        save_dir: Directory to save logs
        epoch: Current epoch
        accumulation_step: Current accumulation step
        total_accumulation_steps: Total gradient accumulation steps
        is_first_batch: Whether this is the first batch of training
    """
    import json
    import os
    from collections import defaultdict
    from pathlib import Path

    import numpy as np
    import torch

    # Create verification directory
    verification_dir = os.path.join(save_dir, "training_verification")
    os.makedirs(verification_dir, exist_ok=True)

    # Create unique filenames for each step instead of overwriting
    step_tracker_file = os.path.join(
        verification_dir,
        f"parameter_tracking_epoch{epoch}_step{accumulation_step}.json",
    )

    # Also maintain a "latest" tracker file for comparing with previous steps
    latest_tracker_file = os.path.join(
        verification_dir, "parameter_tracking_latest.json"
    )

    # State to track
    verification = {
        "epoch": epoch,
        "accumulation_step": accumulation_step,
        "total_steps": total_accumulation_steps,
        "gradients_accumulating": "UNVERIFIED",
        "base_model_updating": "UNVERIFIED",
        "adapter_updating": "UNVERIFIED",
        "timestamp": _make_json_serializable(
            torch.cuda.get_device_properties(0).name
            if torch.cuda.is_available()
            else "cpu"
        ),
    }

    # Track ALL trainable modules and their parameters
    module_stats = defaultdict(
        lambda: {"is_adapter": False, "params": [], "has_changed": False}
    )

    # Function to determine if a module name is likely an adapter
    def is_adapter_module(name):
        adapter_keywords = ["lora", "adapter", "peft", "bias", "prefix"]
        return any(keyword in name.lower() for keyword in adapter_keywords)

    # Collect ALL trainable parameters by module
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Extract module name (first part of parameter path)
                parts = name.split(".")
                module_name = parts[0]
                if len(parts) > 1:
                    # For nested modules, include one more level for better grouping
                    module_name = f"{parts[0]}.{parts[1]}"

                # Determine if this is an adapter parameter
                is_adapter = is_adapter_module(name)

                # Store parameter info
                param_info = {
                    "name": name,
                    "shape": list(param.shape),
                    "norm": float(param.norm().item()),
                    "mean": float(param.mean().item()) if param.numel() > 0 else 0,
                    "std": float(param.std().item()) if param.numel() > 0 else 0,
                    "grad_norm": float(param.grad.norm().item())
                    if param.grad is not None
                    else 0,
                    "first_values": [
                        float(v) for v in param.flatten()[:3].cpu().tolist()
                    ]
                    if param.numel() > 0
                    else [],
                    "abs_sum": float(param.abs().sum().item()),
                }

                module_stats[module_name]["params"].append(param_info)
                module_stats[module_name]["is_adapter"] = (
                    module_stats[module_name]["is_adapter"] or is_adapter
                )

        # Calculate per-module stats
        for module_name, module_data in module_stats.items():
            module_data["param_count"] = len(module_data["params"])
            module_data["total_params"] = sum(
                np.prod(p["shape"]) for p in module_data["params"]
            )
            # Compute the module's total norm for easier comparison
            module_data["total_norm"] = sum(p["norm"] for p in module_data["params"])

    # Store all module stats in verification
    verification["trainable_modules"] = _make_json_serializable(dict(module_stats))

    # Check for gradient accumulation by comparing with previous steps
    if accumulation_step > 0 and os.path.exists(latest_tracker_file):
        try:
            with open(latest_tracker_file, "r") as f:
                prev_data = json.load(f)

            if "trainable_modules" in prev_data:
                prev_modules = prev_data["trainable_modules"]
                gradient_stats = {
                    "modules_with_increased_grad": [],
                    "no_change_modules": [],
                }

                # Compare gradients module by module
                for module_name, module_data in module_stats.items():
                    if module_name in prev_modules:
                        current_grad_sum = sum(
                            p["grad_norm"] for p in module_data["params"]
                        )
                        prev_grad_sum = sum(
                            p["grad_norm"]
                            for p in prev_modules[module_name]["params"]
                            if "grad_norm" in p
                        )

                        if (
                            current_grad_sum > prev_grad_sum + 1e-6
                        ):  # Use small epsilon for floating point comparisons
                            gradient_stats["modules_with_increased_grad"].append(
                                module_name
                            )
                        else:
                            gradient_stats["no_change_modules"].append(module_name)

                verification["gradient_stats"] = gradient_stats
                verification["gradients_accumulating"] = (
                    "YES" if gradient_stats["modules_with_increased_grad"] else "NO"
                )
        except Exception as e:
            verification["gradient_accumulation_error"] = str(e)

    # Check for parameter updates by comparing with initial state
    initial_state_file = os.path.join(verification_dir, "initial_parameters.json")

    # Save initial state if this is the first batch
    if is_first_batch:
        with open(initial_state_file, "w") as f:
            initial_state = {
                "epoch": epoch,
                "trainable_modules": _make_json_serializable(dict(module_stats)),
            }
            json.dump(initial_state, f, indent=2)

    # Compare with initial state if we're past the first few steps
    elif epoch > 0 or accumulation_step > 0:
        if os.path.exists(initial_state_file):
            try:
                with open(initial_state_file, "r") as f:
                    initial_data = json.load(f)

                if "trainable_modules" in initial_data:
                    init_modules = initial_data["trainable_modules"]
                    parameter_changes = {
                        "changed_modules": [],
                        "unchanged_modules": [],
                        "changed_adapter_modules": [],
                        "changed_base_modules": [],
                    }

                    # Track overall stats for adapter vs base model reporting
                    adapter_changed_count = 0
                    adapter_total = 0
                    base_changed_count = 0
                    base_total = 0

                    # Track change thresholds used
                    change_detection = {
                        "norm_threshold": 1e-5,
                        "abs_sum_threshold": 1e-5,
                        "first_value_threshold": 1e-6,
                    }

                    # Check each module for changes
                    for module_name, module_data in module_stats.items():
                        if module_name in init_modules:
                            initial_module = init_modules[module_name]
                            module_changed = False
                            change_metrics = {}

                            # Compare detailed metrics for better change detection
                            # 1. Compare overall module norm
                            norm_diff = abs(
                                module_data["total_norm"] - initial_module["total_norm"]
                            )
                            change_metrics["norm_diff"] = norm_diff

                            # 2. Compare individual parameters
                            param_changes = []
                            for curr_param in module_data["params"]:
                                for init_param in initial_module["params"]:
                                    if curr_param["name"] == init_param["name"]:
                                        # Compare abs_sum for more sensitivity to small changes
                                        abs_sum_diff = abs(
                                            curr_param["abs_sum"]
                                            - init_param["abs_sum"]
                                        )

                                        # Compare first few values directly
                                        first_values_diff = 0
                                        if (
                                            curr_param["first_values"]
                                            and init_param["first_values"]
                                        ):
                                            first_values_diff = sum(
                                                abs(c - i)
                                                for c, i in zip(
                                                    curr_param["first_values"],
                                                    init_param["first_values"],
                                                )
                                            )

                                        param_changed = (
                                            abs_sum_diff
                                            > change_detection["abs_sum_threshold"]
                                            or first_values_diff
                                            > change_detection["first_value_threshold"]
                                            or norm_diff
                                            > change_detection["norm_threshold"]
                                        )

                                        param_changes.append(
                                            {
                                                "name": curr_param["name"],
                                                "abs_sum_diff": abs_sum_diff,
                                                "first_values_diff": first_values_diff,
                                                "changed": param_changed,
                                            }
                                        )

                                        if param_changed:
                                            module_changed = True
                                        break

                            change_metrics["param_changes"] = param_changes
                            module_data["has_changed"] = module_changed
                            module_data["change_metrics"] = change_metrics

                            # Update module lists
                            if module_changed:
                                parameter_changes["changed_modules"].append(module_name)
                                if module_data["is_adapter"]:
                                    parameter_changes["changed_adapter_modules"].append(
                                        module_name
                                    )
                                    adapter_changed_count += 1
                                else:
                                    parameter_changes["changed_base_modules"].append(
                                        module_name
                                    )
                                    base_changed_count += 1
                            else:
                                parameter_changes["unchanged_modules"].append(
                                    module_name
                                )

                            # Update totals
                            if module_data["is_adapter"]:
                                adapter_total += 1
                            else:
                                base_total += 1

                    # Update verification with detailed change information
                    verification["parameter_changes"] = parameter_changes
                    verification["change_detection"] = change_detection

                    # Set top-level verification flags
                    verification["base_model_updating"] = (
                        "YES" if base_changed_count > 0 else "NO"
                    )
                    verification["adapter_updating"] = (
                        "YES" if adapter_changed_count > 0 else "NO"
                    )

                    verification["parameter_change_stats"] = {
                        "base_params_changed": base_changed_count,
                        "total_base_modules": base_total,
                        "adapter_params_changed": adapter_changed_count,
                        "total_adapter_modules": adapter_total,
                        "total_changed_modules": len(
                            parameter_changes["changed_modules"]
                        ),
                        "total_trainable_modules": len(module_stats),
                    }
            except Exception as e:
                verification["parameter_comparison_error"] = str(e)

    # Make the entire structure JSON-serializable
    verification = _make_json_serializable(verification)

    # Save to step-specific file and update the latest tracker
    with open(step_tracker_file, "w") as f:
        json.dump(verification, f, indent=2)

    # Update the latest tracker for the next step to compare against
    with open(latest_tracker_file, "w") as f:
        json.dump(verification, f, indent=2)

    # Create a step-specific summary file
    summary_file = os.path.join(
        verification_dir,
        f"verification_summary_epoch{epoch}_step{accumulation_step}.txt",
    )
    with open(summary_file, "w") as f:
        f.write("======== TRAINING VERIFICATION SUMMARY ========\n\n")
        f.write(
            f"Epoch: {epoch}, Accumulation Step: {accumulation_step + 1}/{total_accumulation_steps}\n\n"
        )

        f.write("CRITICAL VERIFICATION:\n")
        f.write(f"✓ Gradients accumulating: {verification['gradients_accumulating']}\n")
        f.write(f"✓ Base model updating: {verification['base_model_updating']}\n")
        f.write(f"✓ Adapter updating: {verification['adapter_updating']}\n\n")

        # Add detailed module change reports
        if (
            "parameter_changes" in verification
            and "parameter_change_stats" in verification
        ):
            stats = verification["parameter_change_stats"]
            f.write(f"DETAILED MODULE UPDATES:\n")
            f.write(
                f"- Base modules updated: {stats.get('base_params_changed', 0)}/{stats.get('total_base_modules', 0)}\n"
            )
            f.write(
                f"- Adapter modules updated: {stats.get('adapter_params_changed', 0)}/{stats.get('total_adapter_modules', 0)}\n"
            )
            f.write(
                f"- Total modules updated: {stats.get('total_changed_modules', 0)}/{stats.get('total_trainable_modules', 0)}\n\n"
            )

            # List all changed modules
            if verification["parameter_changes"].get("changed_modules"):
                f.write("MODULES WITH CONFIRMED UPDATES:\n")
                for module in verification["parameter_changes"]["changed_modules"]:
                    module_type = (
                        "ADAPTER"
                        if module
                        in verification["parameter_changes"].get(
                            "changed_adapter_modules", []
                        )
                        else "BASE"
                    )
                    f.write(f"- {module} [{module_type}]\n")

            # List all unchanged modules
            if verification["parameter_changes"].get("unchanged_modules"):
                f.write("\nMODULES WITH NO DETECTED UPDATES:\n")
                for module in verification["parameter_changes"]["unchanged_modules"]:
                    is_adapter = any(
                        module == m
                        for m in verification["trainable_modules"]
                        if verification["trainable_modules"][m].get("is_adapter", False)
                    )
                    module_type = "ADAPTER" if is_adapter else "BASE"
                    f.write(f"- {module} [{module_type}]\n")

        # Add change detection thresholds used
        if "change_detection" in verification:
            f.write("\nCHANGE DETECTION THRESHOLDS:\n")
            for metric, threshold in verification["change_detection"].items():
                f.write(f"- {metric}: {threshold}\n")

    # Also update a consolidated "latest" summary file
    latest_summary_file = os.path.join(
        verification_dir, "verification_summary_latest.txt"
    )
    # Copy the step-specific summary to the latest summary
    import shutil

    shutil.copy2(summary_file, latest_summary_file)

    # Log the most important information
    if verification["gradients_accumulating"] != "UNVERIFIED":
        pass

    if (
        verification["base_model_updating"] != "UNVERIFIED"
        and verification["adapter_updating"] != "UNVERIFIED"
    ):
        if "parameter_change_stats" in verification:
            stats = verification["parameter_change_stats"]
            pass

    # If this is the final step in accumulation, create a consolidated report
    if accumulation_step == total_accumulation_steps - 1:
        _create_consolidated_training_verification_report(verification_dir, epoch)

    return verification


def _create_consolidated_training_verification_report(verification_dir, epoch):
    """
    Creates a consolidated report summarizing all verification steps for the epoch.

    This report provides a high-level overview of all the gradient accumulation steps
    and the final parameter update status.

    Args:
        verification_dir: Directory containing verification files
        epoch: Current epoch
    """
    import glob
    import json
    import os

    # Find all step-specific tracker files for this epoch
    tracker_files = sorted(
        glob.glob(
            os.path.join(
                verification_dir, f"parameter_tracking_epoch{epoch}_step*.json"
            )
        )
    )

    if not tracker_files:
        return

    # Create consolidated report
    report_path = os.path.join(
        verification_dir, f"consolidated_verification_epoch{epoch}.txt"
    )

    with open(report_path, "w") as f:
        f.write(
            f"=== CONSOLIDATED TRAINING VERIFICATION REPORT - EPOCH {epoch} ===\n\n"
        )
        f.write(f"Found {len(tracker_files)} verification steps\n\n")

        # Track verification across steps
        step_summaries = []
        adapter_updated = False
        base_model_updated = False

        for tracker_file in tracker_files:
            try:
                with open(tracker_file, "r") as tf:
                    data = json.load(tf)

                step = data.get("accumulation_step", "unknown")
                grads_accum = data.get("gradients_accumulating", "UNKNOWN")
                base_update = data.get("base_model_updating", "UNKNOWN")
                adapter_update = data.get("adapter_updating", "UNKNOWN")

                base_model_updated = base_model_updated or base_update == "YES"
                adapter_updated = adapter_updated or adapter_update == "YES"

                # Get parameter change stats
                if "parameter_change_stats" in data:
                    stats = data["parameter_change_stats"]
                    base_changed = stats.get("base_params_changed", 0)
                    base_total = stats.get("total_base_modules", 0)
                    adapter_changed = stats.get("adapter_params_changed", 0)
                    adapter_total = stats.get("total_adapter_modules", 0)
                else:
                    base_changed = 0
                    base_total = 0
                    adapter_changed = 0
                    adapter_total = 0

                step_summaries.append(
                    {
                        "step": step,
                        "grads_accum": grads_accum,
                        "base_update": base_update,
                        "adapter_update": adapter_update,
                        "base_changed": base_changed,
                        "base_total": base_total,
                        "adapter_changed": adapter_changed,
                        "adapter_total": adapter_total,
                    }
                )
            except Exception as e:
                f.write(f"Error processing {os.path.basename(tracker_file)}: {e}\n")

        # Write step by step summary
        f.write("STEP-BY-STEP VERIFICATION:\n")
        f.write("-------------------------\n")
        for summary in step_summaries:
            step = summary["step"]
            grads = summary["grads_accum"]
            base = summary["base_update"]
            adapter = summary["adapter_update"]

            status_indicators = []
            if grads == "YES":
                status_indicators.append("GRADS✓")
            elif grads == "NO":
                status_indicators.append("GRADS✗")

            if base == "YES":
                status_indicators.append("BASE✓")
            elif base == "NO":
                status_indicators.append("BASE✗")

            if adapter == "YES":
                status_indicators.append("ADAPTER✓")
            elif adapter == "NO":
                status_indicators.append("ADAPTER✗")

            status = " ".join(status_indicators)

            f.write(f"Step {step}: {status}\n")
            if summary["base_changed"] > 0 or summary["adapter_changed"] > 0:
                f.write(
                    f"  - Base modules updated: {summary['base_changed']}/{summary['base_total']}\n"
                )
                f.write(
                    f"  - Adapter modules updated: {summary['adapter_changed']}/{summary['adapter_total']}\n"
                )

        # Overall conclusion
        f.write("\nOVERALL CONCLUSION:\n")
        f.write("------------------\n")
        f.write(
            f"Base model parameters updated: {'YES' if base_model_updated else 'NO'}\n"
        )
        f.write(f"Adapter parameters updated: {'YES' if adapter_updated else 'NO'}\n\n")

        if adapter_updated:
            f.write("✅ TRAINING SUCCESSFUL: Adapter parameters were updated\n")
        else:
            f.write("❌ TRAINING ISSUE: No adapter parameter updates detected\n")

    return report_path


def find_tokenizer_from_model(model):
    """
    Attempt to find a tokenizer from the given model.

    Args:
        model: The model to extract tokenizer from

    Returns:
        The tokenizer if found, None otherwise
    """
    tokenizer = None
    try:
        if hasattr(model, "tokenizer"):
            tokenizer = model.tokenizer
        elif hasattr(model, "model") and hasattr(model.model, "tokenizer"):
            tokenizer = model.model.tokenizer
        elif hasattr(model, "pretrained_model") and hasattr(
            model.pretrained_model, "tokenizer"
        ):
            tokenizer = model.pretrained_model.tokenizer
        elif hasattr(model, "config") and hasattr(model.config, "tokenizer_class"):
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model.config.name_or_path)
        else:
            try:
                from transformers import AutoTokenizer

                # Try to infer tokenizer from model's pretrained name
                if hasattr(model, "config") and hasattr(model.config, "name_or_path"):
                    tokenizer = AutoTokenizer.from_pretrained(model.config.name_or_path)
                else:
                    pass
            except Exception as e:
                pass
    except Exception as e:
        pass

    return tokenizer
