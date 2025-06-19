import gc
import hashlib
import json
import logging
import os
import shutil
import time
import uuid
from copy import deepcopy

import torch
from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from torch.optim import SGD, Adam, AdamW, RMSprop
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import AutoModelForCausalLMWithValueHead
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from mllm.utils.common_imports import *

compute_logger = logging.getLogger("compute_logger")
memory_logger = logging.getLogger("memory_logger")
model_logger = logging.getLogger("model_logger")


class LocalLLM:
    """
    A class that manages HuggingFace and vLLM model instances with multiple adapter configurations.

    This class handles the lifecycle of language models with efficient memory management
    between training and inference phases. It supports both full model fine-tuning and
    parameter-efficient methods like LoRA.

    Args:
        name (str): Identifier for this model instance.
        model_name (str): HuggingFace model identifier.
        device (str): Device to load the model on ('cuda', 'cpu').
        hf_model_init_kwargs (dict): Initialization arguments for HuggingFace model.
        max_model_length (int): Maximum sequence length for processing.
        bits_and_bytes_args (dict, optional): Configuration for quantization. See https://huggingface.co/docs/transformers/v4.51.3/quantization/bitsandbytes
        adapter_configs (dict): Dictionary of adapter configurations.
            Format:
            {
                "adapter_name": {
                    "type": "lora" or "full",
                    "optimizer_method": str,
                    "optimizer_kwargs": dict,
                    "lora_kwargs": dict (for type "lora")
                },
                ...
            }
        generation_args (dict): Text generation parameters.
        eval_with (str): Backend for evaluation ('vllm').
        train_with (str): Backend for training ('hf').
        base_seed (int): Base seed for reproducibility.
        vllm_params (dict): Parameters for vLLM initialization.
        optimizer_on_gpu_during_training (bool): Keep optimizer on GPU during training.
        fully_switch_vllm_weights_after_training (bool): Merge weights to vLLM during training.
        sleep_vllm_during_training (bool): Put vLLM to sleep during training.
        wake_vllm_during_eval (bool): Wake vLLM during evaluation.
        keep_vllm_during_training (bool): Keep vLLM loaded during training.
        keep_hf_during_training (bool): Keep HF model loaded during training.
        keep_hf_during_eval (bool): Keep HF model loaded during evaluation.
        keep_vllm_during_eval (bool): Keep vLLM loaded during evaluation.
        put_hf_on_cpu_during_eval (bool): Put HF model on CPU during evaluation.
        output_directory (str): Directory to save adapters and optimization states.
        export_trained_parameters (bool): Whether to save model parameters after training.
        export_optimizer (bool): Whether to save optimizer state after training.

    Warning:
        - Memory management is critical when using this class, especially with large models.
        - When using full model adapters with vLLM, weight merging is required for evaluation,
          which requires both HF and vLLM models to be loaded simultaneously.
        - vLLM weight merging is NOT compatible with LoRA-enabled vLLM engine instances.
          If both are detected, a warning will be logged and enable_lora will be temporarily
          set to False to allow weight merging to proceed.
    """

    def __init__(
        self,
        name: str = "llama",
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        device: str = "cuda",
        hf_model_init_kwargs={
            "pretrained_model_name_or_path": "meta-llama/Llama-3.1-8B-Instruct",
            "torch_dtype": "bfloat16",
            "device_map": "auto",
            "attn_implementation": "flash_attention_2",
        },
        max_model_length=8000,
        bits_and_bytes_args=None,
        adapter_configs=None,
        generation_args={
            "max_new_tokens": 300,
            "do_sample": True,
            "temperature": 1.0,
            "top_k": 0,
            "top_p": 1.0,
            "repetition_penalty": 1.0,
        },
        eval_with="vllm",
        train_with="hf",
        base_seed: int = 42,
        vllm_params={},
        restrict_tokens=None,
        optimizer_on_gpu_during_training=True,
        fully_switch_vllm_weights_after_training=False,
        sleep_vllm_during_training=True,
        wake_vllm_during_eval=True,
        keep_vllm_during_training=True,
        keep_vllm_during_eval=True,
        put_hf_on_cpu_during_eval=False,
        output_directory=None,
        export_trained_parameters=True,
        export_optimizer=True,
        adapter_names=None,  # not used, for backward compatibility
    ) -> None:
        """
        Initializes the LocalLLM.
        """
        super().__init__()
        self.output_directory = output_directory
        self.vllm_params = vllm_params
        self.name = name
        self.device = torch.device(device) if device else torch.device("cuda")
        self.model_name = model_name
        self.hf_model_init_kwargs = hf_model_init_kwargs
        self.max_model_length = max_model_length
        self.restrict_tokens = restrict_tokens

        # Tracking current state
        self.current_adapter_name = None
        self.current_vllm_adapter_name = None

        self.adapter_configs = adapter_configs
        self.adapter_names = self.adapter_configs.keys()

        # Path management / imports
        self.adapter_paths = {}
        for adapter_name in self.adapter_names:
            adapter_config = self.adapter_configs[adapter_name]
            adapter_path = os.path.join(self.output_directory, adapter_name, "model")
            hf_server_import_kwargs = adapter_config.get(
                "hf_server_import_kwargs", None
            )
            local_import_adapter_path = adapter_config.get(
                "local_import_adapter_path", None
            )
            if hf_server_import_kwargs is not None:
                # dealing with hf shenanigans again here - sorry
                model_logger.info(
                    f"Downloading adapter {adapter_name}\
                    from {hf_server_import_kwargs}"
                )
                from huggingface_hub import snapshot_download

                snapshot_download(**hf_server_import_kwargs, local_dir=adapter_path)
                additional_path = hf_server_import_kwargs.get("allow_patterns")[0]
                additional_path = additional_path[:-2]  # remove /model
                adapter_path = os.path.join(adapter_path, additional_path)
            elif local_import_adapter_path is not None:
                model_logger.info(
                    f"Copying adapter {adapter_name}\
                     from {local_import_adapter_path} to {adapter_path}"
                )
                shutil.copytree(
                    src=local_import_adapter_path,
                    dst=adapter_path,
                    dirs_exist_ok=True,
                )
            self.adapter_paths[adapter_name] = adapter_path

        # Copy external adapters to local directory (ensures we don't modify the original ones)

        self.optimizer_paths = {
            adapter_name: os.path.join(
                self.output_directory, adapter_name, "optimizer", "optimizer.pt"
            )
            for adapter_name in self.adapter_names
        }

        # ID management for tracking adapter versions
        self.adapter_train_ids = {
            adapter_name: self.short_id_generator()
            for adapter_name in self.adapter_names
        }

        self.adapter_eval_ids = deepcopy(self.adapter_train_ids)
        self.vllm_loaded_adapter_versions = deepcopy(self.adapter_train_ids)

        # Feature detection
        self.at_least_one_full_adapter = any(
            config["type"] == "full" for config in self.adapter_configs.values()
        )

        # Check if we have LoRA adapters but export_trained_parameters is False
        has_lora_adapters = any(
            config["type"] == "lora" for config in self.adapter_configs.values()
        )

        if has_lora_adapters and not export_trained_parameters:
            model_logger.warning(
                "LoRA adapters detected but export_trained_parameters is set to False. "
                "Setting export_trained_parameters to True to ensure LoRA weights are saved."
            )
            export_trained_parameters = True

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Setup padding token to be same as EOS token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Configure sampling parameters
        self.hf_sampling_params = generation_args
        self.vllm_sampling_params = SamplingParams(
            temperature=generation_args["temperature"],
            top_k=-1 if generation_args["top_k"] == 0.0 else generation_args["top_k"],
            top_p=generation_args["top_p"],
            max_tokens=generation_args["max_new_tokens"],
            repetition_penalty=generation_args["repetition_penalty"],
        )

        self.optimizer_on_gpu_during_training = optimizer_on_gpu_during_training
        self.switch_weights_vllm_before_eval = fully_switch_vllm_weights_after_training
        self.sleep_vllm_during_training = sleep_vllm_during_training
        self.wake_vllm_during_eval = wake_vllm_during_eval
        self.keep_vllm_during_training = keep_vllm_during_training
        self.keep_vllm_during_eval = keep_vllm_during_eval
        self.put_hf_on_cpu_during_eval = put_hf_on_cpu_during_eval
        self.train_with = train_with
        self.eval_with = eval_with
        self.export_trained_parameters = export_trained_parameters
        self.export_optimizer = export_optimizer

        self.hf_model = None
        self.vllm_model = None
        self.peft_model = None  # Single PeftModel instance to hold all adapters

        self.optimizer = None
        self.current_lora_request = None
        self.adapter_configs = adapter_configs
        self.adapter_names = self.adapter_configs.keys()
        self.active_adapters = {
            adapter_name: False for adapter_name in self.adapter_names
        }

        # Extract configurations from adapter configs
        self.adapter_types = {
            name: config["type"] for name, config in self.adapter_configs.items()
        }

        self.optimizer_methods = {
            name: config["optimizer_method"]
            for name, config in self.adapter_configs.items()
        }

        self.optimizer_kwargs = {
            name: config["optimizer_kwargs"]
            for name, config in self.adapter_configs.items()
        }

        self.lora_kwargs = {
            name: config["lora_kwargs"] for name, config in self.adapter_configs.items()
        }

        self.base_seed = base_seed

        # Setup quantization if provided
        if isinstance(bits_and_bytes_args, dict):
            self.hf_quantization_config = BitsAndBytesConfig(**bits_and_bytes_args)
        else:
            self.hf_quantization_config = None

    def prepare_adapter_train(self, adapter_name: str):
        """
        Prepares the model for training with the specified adapter.

        This method handles the memory management between vLLM and HuggingFace models,
        loads the appropriate adapter configuration, and sets up the optimizer.

        Args:
            adapter_name (str): Name of the adapter to prepare for training.

        Notes:
            - This method may unload vLLM to free memory if keep_vllm_during_training is False.
            - A new adapter ID is generated to track changes in the adapter during training.
            - For full model training, all parameters are set to requires_grad=True.
            - For LoRA training, only adapter parameters are trainable.

        Warning:
            This method changes the active adapter and may affect memory usage significantly.
            If using a LoRA adapter, the weights need to be saved externally.
        """

        assert (
            adapter_name in self.adapter_names
        ), f"Adapter {adapter_name} not found in {self.adapter_names}."

        # Free memory if needed
        if not self.keep_vllm_during_training:
            self.vllm_model = None
            gc.collect()
            torch.cuda.empty_cache()

        if self.sleep_vllm_during_training and self.vllm_model is not None:
            self.vllm_model.sleep()

        # Create new model version with a new ID
        self.adapter_train_ids[adapter_name] = self.short_id_generator()

        self.log_gpu_usage(
            f"Before loading HF model with adapter {adapter_name} for training."
        )

        model_logger.info(f"Preparing adapter {adapter_name} for training.")

        start_time = time.time()
        self.current_adapter_name = adapter_name
        adapter_path = self.adapter_paths[self.current_adapter_name]
        adapter_type = self.adapter_types[self.current_adapter_name]

        # Initialize base model if needed
        new_hf_model = False
        if self.hf_model is None:
            new_hf_model = True
            if self.hf_quantization_config is None:
                # Avoids HF bugs with pre-existing quant. conflicts in base model
                self.hf_model = AutoModelForCausalLM.from_pretrained(
                    pretrained_model_name_or_path=self.model_name,
                    **self.hf_model_init_kwargs,
                )
            else:
                self.hf_model = AutoModelForCausalLM.from_pretrained(
                    pretrained_model_name_or_path=self.model_name,
                    quantization_config=self.hf_quantization_config,
                    **self.hf_model_init_kwargs,
                )

        # Full model training
        if adapter_type == "full":
            model_logger.info(f"Setting up full model adapter for {adapter_name}")
            for param in self.hf_model.parameters():
                param.requires_grad = True
            param_groups = [
                {
                    "params": self.hf_model.parameters(),
                    "lr": self.optimizer_kwargs[adapter_name].get("lr", 1e-5),
                }
            ]
            self.optimizer = getattr(
                __import__(
                    "torch.optim", fromlist=[self.optimizer_methods[adapter_name]]
                ),
                self.optimizer_methods[adapter_name],
            )(param_groups, **self.optimizer_kwargs[adapter_name])

        # Prepare for LoRA training
        elif adapter_type == "lora":
            model_logger.info(f"Setting up LoRA adapter for {adapter_name}")

            if new_hf_model:
                model_logger.info(
                    f"Initializing PEFT model with LoRA adapter {adapter_name}"
                )
                lora_config = LoraConfig(**self.lora_kwargs[adapter_name])
                self.hf_model = get_peft_model(
                    model=self.hf_model,
                    peft_config=lora_config,
                    adapter_name=adapter_name,
                )

            # if self.hf_quantization_config is not None and not hasattr(
            #     self.hf_model, "is_quantized"
            # ):
            #     self.hf_model = prepare_model_for_kbit_training(self.hf_model)
            #     self.hf_model.is_quantized = True

            if adapter_name not in getattr(self.hf_model, "peft_config", {}):
                model_logger.info(
                    f"Adding new adapter {adapter_name} to existing PEFT model"
                )
                lora_config = LoraConfig(**self.lora_kwargs[adapter_name])
                self.hf_model.add_adapter(
                    adapter_name=adapter_name, peft_config=lora_config
                )
            self.hf_model.set_adapter(adapter_name)
            trainable_params = [
                p for p in self.hf_model.parameters() if p.requires_grad
            ]
            optimizer_class = getattr(
                __import__(
                    "torch.optim", fromlist=[self.optimizer_methods[adapter_name]]
                ),
                self.optimizer_methods[adapter_name],
            )
            self.optimizer = optimizer_class(
                trainable_params, **self.optimizer_kwargs[adapter_name]
            )

        self.hf_model.train()

        trainable_params = [p for p in self.hf_model.parameters() if p.requires_grad]

        optimizer_path = self.optimizer_paths[adapter_name]
        if os.path.exists(optimizer_path):
            model_logger.info(f"Loading optimizer state from {optimizer_path}")
            try:
                self.optimizer.load_state_dict(torch.load(optimizer_path))
            except ValueError as e:
                model_logger.warning(f"Failed to load optimizer state: {e}")
                model_logger.warning("Creating a new optimizer state instead.")
                # Continue with the new optimizer that was already created

        # Log model statistics
        total_params = sum(p.numel() for p in self.hf_model.parameters())
        trainable_params = sum(
            p.numel() for p in self.hf_model.parameters() if p.requires_grad
        )

        model_logger.info(f"Total Parameters: {total_params}")
        model_logger.info(
            f"Trainable Parameters: {trainable_params} "
            f"({trainable_params/total_params:.2%})"
        )

        compute_logger.info(
            f"HF model loading time: {time.time() - start_time:.2f} seconds."
        )

        self.log_gpu_usage(
            f"After loading HF model with adapter {adapter_name} for training."
        )

    def prepare_adapter_eval(self, adapter_name: str, seed_offset: int = 0):
        """
        Prepares the model for evaluation with the specified adapter.

        This method handles memory management between HuggingFace and vLLM models,
        loads the appropriate adapter, and sets up the environment for inference.

        Args:
            adapter_name (str): Name of the adapter to use for evaluation.
            seed_offset (int, optional): Offset to apply to the base seed for deterministic generation.

        Notes:
            - This method may unload the HF model if keep_hf_during_eval is False.
            - For full model adapters, weights are merged into vLLM when necessary.
            - For LoRA adapters, a LoRA request is set up for vLLM.

        Warning:
            - Weight merging requires both HF and vLLM models to be loaded, which can be memory intensive.
            - VLLM weight merging is NOT compatible with LoRA-enabled vLLM engine instances.
              If both are detected, a warning will be logged and enable_lora will be temporarily
              set to False to allow weight merging to proceed.
            - If HF model is not loaded when weight merging is required, a warning will be logged
              and weight merging will be skipped.
        """
        assert (
            adapter_name in self.adapter_names
        ), f"Adapter {adapter_name} not found in {self.adapter_names}."

        model_logger.info(f"Preparing adapter {adapter_name} for evaluation.")
        adapter_type = self.adapter_types[adapter_name]
        adapter_path = self.adapter_paths[adapter_name]
        if self.hf_model is not None:
            self.hf_model.eval()
        # Track adapter changes
        self.current_adapter_name = adapter_name

        is_adapter_changed = self.current_adapter_name != self.current_vllm_adapter_name

        is_adapter_updated = (
            self.vllm_loaded_adapter_versions[self.current_adapter_name]
            != self.adapter_train_ids[self.current_adapter_name]
        )
        if is_adapter_updated:
            model_logger.info(
                f"Adapter {adapter_name} has been updated. vLLM will be updated."
            )

        is_full_adapter = adapter_type == "full"

        weight_switch_required = (
            is_adapter_changed or is_adapter_updated
        ) and self.switch_weights_vllm_before_eval

        if self.eval_with == "vllm":
            vllm_params = self.vllm_params
            # vLLM extension for weight switch.
            worker_extension_cls = None
            hf_model = self.hf_model

            # Prepare for weight merging
            if weight_switch_required:
                if self.hf_model is None:
                    model_logger.warning(
                        f"HF model is not loaded. Cannot merge weights for {adapter_name}."
                    )
                    weight_switch_required = False

                # V1 version of vLLM is not compatible with weight switch.
                os.environ["VLLM_USE_V1"] = "0"
                worker_extension_cls = (
                    "models.updatable_worker.UpdatableWorkerExtension"
                )

                # Instead of assertion, check and log a warning if enable_lora is True
                if self.vllm_params.get("enable_lora", False):
                    model_logger.warning(
                        "VLLM weight merging is not supported with LoRA-enabled "
                        "vLLM engine instances. Temporarily setting enable_lora to False."
                    )
                    # Make a copy of vllm_params and modify enable_lora
                    vllm_params = dict(self.vllm_params)
                    vllm_params["enable_lora"] = False

            # Initialize vLLM if not loaded
            if self.vllm_model is None:
                self.log_gpu_usage(f"Before loading VLLM model with {adapter_name}.")
                start_time = time.time()

                if self.switch_weights_vllm_before_eval:
                    self.vllm_model = LLM(
                        self.model_name,
                        **vllm_params,
                        worker_extension_cls=worker_extension_cls,
                    )
                else:
                    self.vllm_model = LLM(
                        self.model_name,
                        **vllm_params,
                    )

                compute_logger.info(
                    f"VLLM model loading time: {time.time() - start_time:.2f} seconds."
                )
                self.log_gpu_usage(f"After loading VLLM model with {adapter_name}.")

            if self.wake_vllm_during_eval:
                self.vllm_model.wake_up()

            # Update tracking information
            self.current_vllm_adapter_name = adapter_name
            self.vllm_loaded_adapter_versions[adapter_name] = self.adapter_train_ids[
                adapter_name
            ]

            # Handle weight merging for vLLM if required
            if weight_switch_required and self.hf_model is not None:
                if adapter_type == "full":
                    model_logger.info(
                        f"Merging weights for {adapter_name} to vLLM model."
                    )
                    self.hf_model.eval()
                    if adapter_type == "lora":
                        model_logger.error(
                            "LoRA weight merging is not supported for vLLM."
                        )

                    model_logger.info(
                        "Using direct weight update for full model adapter"
                    )
                    for name, param in hf_model.named_parameters():
                        self.vllm_model.collective_rpc(
                            "update_weight", args=(name, param.data)
                        )

            # Use vLLM LoRA Request instead of full weight switch for LoRA adapters
            elif (
                adapter_type == "lora"
                and os.path.exists(adapter_path)
                and not self.switch_weights_vllm_before_eval
            ):
                model_logger.info(f"Creating LoRA request for adapter {adapter_name}")
                self.current_lora_request = LoRARequest(
                    adapter_name, self.adapter_train_ids[adapter_name], adapter_path
                )
            else:
                self.current_lora_request = None
        else:
            model_logger.error(f"Unsupported evaluation method: {self.eval_with}")

    def prompt(self, contexts) -> list:
        """
        Generates responses from the model based on the provided contexts.

        Args:
            contexts (List[dict]): Chat contexts for generation.
            self.restrict_tokens (List[str], optional): List of allowed string tokens. If provided, restricts generation to only these tokens.

        Returns:
            list: List of generated responses from the model.

        Notes:
            - This method only supports vLLM for text generation.
            - Empty contexts list will return an empty list.
            - The method uses the currently active adapter.
        """
        if not contexts:
            return []

        # Apply chat template to contexts
        texts = self.tokenizer.apply_chat_template(
            contexts, tokenize=False, add_generation_prompt=True
        )

        sampling_params = self.vllm_sampling_params
        # If self.restrict_tokens is provided, convert to token ids and set allowed_token_ids
        if self.restrict_tokens is not None:
            allowed_token_ids = []
            for token in self.restrict_tokens:
                # Tokenize and take the first token id (assume single-token for each string)
                token_ids = self.tokenizer(token, add_special_tokens=False)["input_ids"]
                if len(token_ids) != 1:
                    model_logger.warning(
                        f"Token '{token}' does not map to a single token. Skipping."
                    )
                    continue
                allowed_token_ids.append(token_ids[0])
            if not allowed_token_ids:
                model_logger.error(
                    "No valid allowed_token_ids found for self.restrict_tokens. Generation will not be restricted."
                )
            else:
                # Clone sampling_params and set allowed_token_ids
                sampling_params = sampling_params.clone()
                sampling_params.allowed_token_ids = allowed_token_ids

        if self.eval_with == "vllm":
            # Disable deterministic seed for more diversity
            sampling_params.seed = None

            if self.current_lora_request is not None:
                model_logger.info(
                    f"Generating using VLLM with LoRA. "
                    f"Current LoRA request ID is {self.current_lora_request.adapter_id}. "
                    f"Current LoRA adapter path is {self.current_lora_request.path}."
                )

            # Generate responses
            decoded = self.vllm_model.generate(
                texts,
                sampling_params=sampling_params,
                lora_request=self.current_lora_request,
            )
            responses = [d.outputs[0].text for d in decoded]
            del decoded
        else:
            model_logger.error(f"Unsupported generation method: {self.eval_with}")
            return []

        return responses

    def export_current_adapter_and_optimizer(self) -> None:
        """
        Exports the current adapter to the configured output directory.
        Also exports the optimizer state to
        the output directory if self.export_optimizer is True.
        """
        adapter_path = self.adapter_paths[self.current_adapter_name]
        os.makedirs(adapter_path, exist_ok=True)
        self.export_adapter(self.current_adapter_name, adapter_path)
        if self.export_optimizer:
            optimizer_state_path = self.optimizer_paths[self.current_adapter_name]
            optimizer_dir = os.path.dirname(optimizer_state_path)
            os.makedirs(optimizer_dir, exist_ok=True)
            torch.save(self.optimizer.state_dict(), optimizer_state_path)
            model_logger.info(f"Optimizer state saved to {optimizer_state_path}")

    def checkpoint_all_adapters(self, checkpoint_indicator: str) -> None:
        """
        Checkpoints all adapters to the configured output directory.
        """
        for adapter_name in self.adapter_names:
            output_dir = os.path.join(self.output_directory, "checkpoints")
            os.makedirs(output_dir, exist_ok=True)
            date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            export_path = os.path.join(
                output_dir, f"{adapter_name}-{checkpoint_indicator}-{date_str}"
            )
            self.export_adapter(adapter_name, export_path)

    def export_adapter(self, adapter_name: str, export_path: str) -> None:
        """
        Exports the current adapter to the configured output directory.

        If checkpoint is True, the adapter will be saved as a checkpoint in a safe folder, not at
        the adapter path.

        For 'full' adapters, the entire model is saved.
        For 'lora' adapters, only the LoRA parameters are saved.
        The optimizer state is also saved if configured.

        Notes:
            - Directory structure is organized by adapter name.
            - Export can be disabled via export_trained_parameters and export_optimizer flags.

        Warning:
            This operation may take significant disk space for full model adapters.
        """

        adapter_type = self.adapter_types[adapter_name]

        # Create output directory if it doesn't exist
        os.makedirs(export_path, exist_ok=True)

        # Save model parameters
        if self.export_trained_parameters:
            if adapter_type == "full":
                model_logger.info(f"Saving full model to {export_path}")
                self.hf_model.save_pretrained(export_path)
            elif adapter_type == "lora":
                model_logger.info(f"Saving LoRA adapter to {export_path}")

                # Properly save PEFT adapter
                if isinstance(self.hf_model, PeftModel):
                    # Ensure the current adapter is active
                    if adapter_name != self.hf_model.active_adapter:
                        model_logger.info(
                            f"Setting active adapter to {adapter_name} for saving"
                        )
                        self.hf_model.set_adapter(adapter_name)

                    # The following code is because HF creates different sub folders with
                    # with different configs.. TODO: fix this efficiently
                    temp_export_path = os.path.join(export_path, "_temp")
                    os.makedirs(temp_export_path, exist_ok=True)
                    self.hf_model.save_pretrained(
                        temp_export_path,
                        adapter_name=adapter_name,
                        safe_serialization=True,
                    )
                    subfolder_path = os.path.join(temp_export_path, adapter_name)
                    if os.path.exists(subfolder_path):
                        for item in os.listdir(subfolder_path):
                            src = os.path.join(subfolder_path, item)
                            dst = os.path.join(export_path, item)
                            shutil.move(src, dst)
                    # Clean up temp directory
                    shutil.rmtree(temp_export_path)

                else:
                    model_logger.warning(
                        "Model is not a PeftModel instance. Unable to save LoRA adapter."
                    )

    def log_gpu_usage(self, message: str) -> None:
        """
        Logs current GPU memory usage.

        Args:
            message (str): A message to include in the log entry.
        """
        free_memory, total_memory = torch.cuda.mem_get_info()
        gpu_memory = (total_memory - free_memory) / (1024**3)
        memory_logger.info(f"{message}: GPU memory allocated: {gpu_memory:.2f} GB")

    def short_id_generator(self) -> int:
        """
        Generates a short unique ID for tracking adapter versions.

        Returns:
            int: An 8-digit integer ID.
        """
        return int(str(uuid.uuid4().int)[:8])
