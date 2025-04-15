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
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import AutoModelForCausalLMWithValueHead
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from torch.optim import Adam, AdamW, SGD, RMSprop
from utils.common_imports import *

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
        bits_and_bytes_args (dict, optional): Configuration for quantization.
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
        optimizer_on_gpu_during_training=True,
        fully_switch_vllm_weights_after_training=False,
        sleep_vllm_during_training=False,
        wake_vllm_during_eval=True,
        keep_vllm_during_training=False,
        keep_hf_during_training=True,
        keep_hf_during_eval=False,
        keep_vllm_during_eval=True,
        output_directory=None,
        export_trained_parameters=True,
        export_optimizer=True,
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

        # Tracking current state
        self.current_adapter_name = None
        self.current_vllm_adapter_name = None

        self.adapter_configs = adapter_configs
        self.adapter_names = self.adapter_configs.keys()
        
        # Path management
        self.adapter_paths = {
            adapter_name: os.path.join(self.output_directory, adapter_name, "model")
            for adapter_name in self.adapter_names
        }
        
        self.optimizer_paths = {
            adapter_name: os.path.join(
                self.output_directory, 
                adapter_name, 
                "optimizer", 
                "optimizer.pt"
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
            config["type"] == "full" 
            for config in self.adapter_configs.values()
        )

        # Check if we have LoRA adapters but export_trained_parameters is False
        has_lora_adapters = any(
            config["type"] == "lora"
            for config in self.adapter_configs.values()
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
        
        # Setup quantization if provided
        self.bits_and_bytes_configs = (
            BitsAndBytesConfig(**bits_and_bytes_args) 
            if bits_and_bytes_args else None
        )
        
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
        self.keep_hf_during_training = keep_hf_during_training
        self.keep_hf_during_eval = keep_hf_during_eval
        self.keep_vllm_during_eval = keep_vllm_during_eval
        self.train_with = train_with
        self.eval_with = eval_with
        self.export_trained_parameters = export_trained_parameters
        self.export_optimizer = export_optimizer

        self.hf_model = None
        self.vllm_model = None

        self.current_optimizer = None
        self.current_lora_request = None
        self.adapter_configs = adapter_configs
        self.adapter_names = self.adapter_configs.keys()
        self.active_adapters = {
            adapter_name: False for adapter_name in self.adapter_names
        }
        
        # Extract configurations from adapter configs
        self.adapter_types = {
            name: config["type"] 
            for name, config in self.adapter_configs.items()
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
            name: config["lora_kwargs"] 
            for name, config in self.adapter_configs.items()
        }

        self.base_seed = base_seed

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

        assert adapter_name in self.adapter_names, \
            f"Adapter {adapter_name} not found in {self.adapter_names}."

        # Free memory if needed
        if not self.keep_vllm_during_training:
            self.vllm_model = None
            gc.collect()
            torch.cuda.empty_cache()

        if self.sleep_vllm_during_training and self.vllm_model is not None:
            self.vllm_model.sleep()

        # Create new model version with a new ID
        previous_id = self.adapter_train_ids[adapter_name]
        self.adapter_train_ids[adapter_name] = self.short_id_generator()
        
        self.log_gpu_usage(
            f"Before loading HF model with adapter {adapter_name} for training."
        )
        
        model_logger.info(f"Preparing adapter {adapter_name} for training.")

        start_time = time.time()
        self.current_adapter_name = adapter_name
        adapter_path = self.adapter_paths[self.current_adapter_name]
        adapter_type = self.adapter_types[self.current_adapter_name]

        # Load model if needed
        if self.hf_model is None:
            if os.path.exists(adapter_path) and adapter_type == "full":
                model_logger.info(f"Loading HF model from {adapter_path} for training.")
                self.hf_model = AutoModelForCausalLM.from_pretrained(
                    pretrained_model_name_or_path=adapter_path,
                    **self.hf_model_init_kwargs,
                )
            else: 
                model_logger.info(f"Loading HF model named {self.model_name} for training.")
                self.hf_model = AutoModelForCausalLM.from_pretrained(
                    pretrained_model_name_or_path=self.model_name,
                    **self.hf_model_init_kwargs,
                )
            
        # Configure model for training based on adapter type
        if adapter_type == "full":
            model_logger.info("Setting up full model training (no LoRA)")
            for param in self.hf_model.parameters():
                param.requires_grad = True
                
        elif adapter_type == "lora":
            adapter_id = str(self.adapter_train_ids[adapter_name])

            # Clean up any existing adapters first to avoid parameter bloat
            for id in list(self.adapter_train_ids.values()) + [previous_id]:
                try: 
                    self.hf_model.delete_adapter(str(id))
                    model_logger.info(f"Deleted adapter {id}")
                except: pass
            

            if os.path.exists(adapter_path):
                model_logger.info(f"Loading HF LoRA adapter from {adapter_path} for training.")
                self.hf_model.load_adapter(
                    peft_model_id=adapter_path, 
                    adapter_name=adapter_id
                )

            else:
                model_logger.info(f"Creating new HF LoRA adapter for {adapter_name} for training.")
                lora_config = LoraConfig(**self.lora_kwargs[adapter_name])
                self.hf_model.add_adapter(
                    adapter_name=adapter_id,
                    adapter_config=lora_config
                )

            self.hf_model.set_adapter(adapter_name=adapter_id)

        self.hf_model.train()
            
        # Initialize and load optimizer
        optimizer_class = getattr(
            __import__(
                'torch.optim', 
                fromlist=[self.optimizer_methods[adapter_name]]
            ), 
            self.optimizer_methods[adapter_name]
        )
        
        self.current_optimizer = optimizer_class(
            self.hf_model.parameters(),
            **self.optimizer_kwargs[adapter_name]
        )
        
        optimizer_path = self.optimizer_paths[adapter_name]
        if os.path.exists(optimizer_path):
            model_logger.info(f"Loading optimizer state from {optimizer_path}")
            try:
                self.current_optimizer.load_state_dict(
                    torch.load(optimizer_path)
                )
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
        assert adapter_name in self.adapter_names, \
            f"Adapter {adapter_name} not found in {self.adapter_names}."

        model_logger.info(f"Preparing adapter {adapter_name} for evaluation.")
        adapter_type = self.adapter_types[adapter_name]
        adapter_path = self.adapter_paths[adapter_name]
        
        # Track adapter changes
        self.current_adapter_name = adapter_name
        
        is_adapter_changed = (
            self.current_adapter_name != self.current_vllm_adapter_name
        )
        
        is_adapter_updated = (
            self.vllm_loaded_adapter_versions[self.current_adapter_name] 
            != self.adapter_train_ids[self.current_adapter_name]
        )
        
        is_full_adapter = adapter_type == "full"
        
        weight_switch_required = (
            (is_adapter_changed or is_adapter_updated) 
            and self.switch_weights_vllm_before_eval
        )

        # Free memory if needed
        if not self.keep_hf_during_eval:
            model_logger.info(f"Deleting HF model for evaluation with {adapter_name}.")
            del self.hf_model
            gc.collect()
            torch.cuda.empty_cache()
            self.hf_model = None

        if self.eval_with == "vllm":

            vllm_params = self.vllm_params
            # vLLM extension for weight switch.
            worker_extension_cls = 'models.updatable_worker.UpdatableWorkerExtension'
            hf_model = self.hf_model

            # Prepare 
            if weight_switch_required:

                if self.hf_model is None:
                    model_logger.warning(f"HF model is not loaded. Cannot merge weights for {adapter_name}.")

                # V1 version of vLLM is not compatible with weight switch.
                os.environ["VLLM_USE_V1"] = '0' 
                

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

                self.vllm_model = LLM(
                    self.model_name,
                    **vllm_params,
                    worker_extension_cls=worker_extension_cls
                )
                    
                compute_logger.info(
                    f"VLLM model loading time: {time.time() - start_time:.2f} seconds."
                )
                self.log_gpu_usage(f"After loading VLLM model with {adapter_name}.")

            if self.wake_vllm_during_eval:
                self.vllm_model.wake_up()

            # Update tracking information
            self.current_vllm_adapter_name = adapter_name
            self.vllm_loaded_adapter_versions[adapter_name] = (
                self.adapter_train_ids[adapter_name]
            )

            # Handle full model weight switch
            if weight_switch_required and hf_model is not None:
                if os.path.exists(adapter_path):
                    model_logger.info(
                        f"Merging weights for {adapter_name} from {adapter_path} to vLLM model."
                    )
                    hf_model.eval()
                    
                    # Check adapter type
                    if adapter_type == "full":
                        # For full model adapters, update weights directly
                        model_logger.info("Using direct weight update for full model adapter")
                        for name, param in hf_model.named_parameters():
                            # Create a copy of the parameter data to avoid modifying the original
                            self.vllm_model.collective_rpc(
                                'update_weight', 
                                args=(name, param.data)
                            )
                    elif adapter_type == "lora":
                        # For LoRA adapters, we need to properly handle the delta weights
                        model_logger.info("Using LoRA-aware weight update for LoRA adapter")
                        
                        # Collect all LoRA modules for easy lookup
                        lora_modules = {}
                        for name, module in hf_model.named_modules():
                            if hasattr(module, 'get_delta_weight') and hasattr(module, 'active_adapter'):
                                lora_modules[name] = module
                        
                        # Track parameters we've already processed
                        processed_params = set()
                        
                        # Process all modules with get_delta_weight method (LoRA modules)
                        for module_name, module in lora_modules.items():
                            if hasattr(module, 'weight'):
                                param_name = f"{module_name}.weight"
                                
                                # Make a copy of the base weight to avoid modifying the original
                                base_weight = module.weight.data
                                
                                # Get delta from all active adapters
                                for adapter in module.active_adapter:
                                    delta = module.get_delta_weight(adapter)
                                    # Apply delta to get the full weight (using the copy)
                                    combined_weight = base_weight + delta
                                    
                                    # Update weight in vLLM
                                    self.vllm_model.collective_rpc(
                                        'update_weight', 
                                        args=(param_name, combined_weight)
                                    )
                                    
                                    # Mark as processed
                                    processed_params.add(param_name)

            # Use vLLM LoRA Request instead of full weight switch
            elif adapter_type == "lora" and os.path.exists(adapter_path) and not self.switch_weights_vllm_before_eval:
                self.current_lora_request = LoRARequest(
                    adapter_name, 
                    self.adapter_train_ids[adapter_name], 
                    adapter_path
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
            contexts, 
            tokenize=False, 
            add_generation_prompt=True
        )


        if self.eval_with == "vllm":
            # Disable deterministic seed for more diversity
            self.vllm_sampling_params.seed = None

            if self.current_lora_request is not None:
                model_logger.info(
                    f"Generating using VLLM with LoRA. "
                    f"Current LoRA request ID is {self.current_lora_request.adapter_id}. "
                    f"Current LoRA adapter path is {self.current_lora_request.path}."
                )
                
            # Generate responses
            decoded = self.vllm_model.generate(
                texts,
                sampling_params=self.vllm_sampling_params,
                lora_request=self.current_lora_request,
            )
            responses = [d.outputs[0].text for d in decoded]
            del decoded
        else:
            model_logger.error(f"Unsupported generation method: {self.eval_with}")
            return []


        return responses

    def export_current_adapter(self) -> None:
        """
        Exports the current adapter to the configured output directory.
        
        For 'full' adapters, the entire model is saved.
        For 'lora' adapters, only the LoRA parameters are saved.
        The optimizer state is also saved if configured.
        
        Notes:
            - Directory structure is organized by adapter name.
            - Export can be disabled via export_trained_parameters and export_optimizer flags.
            
        Warning:
            This operation may take significant disk space for full model adapters.
        """


        # Save model parameters
        if self.export_trained_parameters:
            adapter_path = self.adapter_paths[self.current_adapter_name]
            adapter_type = self.adapter_types[self.current_adapter_name]
            os.makedirs(adapter_path, exist_ok=True)
            if adapter_type == "full":
                model_logger.info(f"Saving full model to {adapter_path}")
            elif adapter_type == "lora":
                model_logger.info(f"Saving LoRA weights to {adapter_path}")
                
            self.hf_model.save_pretrained(adapter_path)
            
        # Save optimizer state
        if self.export_optimizer:
            optimizer_state_path = self.optimizer_paths[self.current_adapter_name]
            optimizer_dir = os.path.dirname(optimizer_state_path)
            os.makedirs(optimizer_dir, exist_ok=True)
            
            torch.save(
                self.current_optimizer.state_dict(), 
                optimizer_state_path
            )
            
            model_logger.info(
                f"Optimizer state saved to {optimizer_state_path}"
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

        
