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
import vllm
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from utils.common_imports import *

compute_logger = logging.getLogger("compute_logger")
memory_logger = logging.getLogger("memory_logger")
model_logger = logging.getLogger("model_logger")


class LeanLocalLLM:
    """
    
    """

    def __init__(
        self,
        name: str = "llama",
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        device: str = "cuda",
        hf_model_init_kwargs: dict = {},
        max_model_length: int = 8000,
        bits_and_bytes_args: dict = None,
        generation_args={
            "max_new_tokens": 300,
            "do_sample": True,
            "temperature": 1.0,
            "top_k": 0,
            "top_p": 1.0,
            "repetition_penalty": 1.0,
        },
        base_seed: int = 42,
        vllm_params: dict = {},
        adapter_configs: list[dict] =[{}],
        restrict_tokens=None,
        output_directory=None,
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

        # Path management / imports
        self.adapter_paths = {}
        for adapter_name in self.adapter_names:
            adapter_config = self.adapter_configs[adapter_name]
            adapter_path = os.path.join(self.output_directory, adapter_name, "model")
            self.adapter_paths[adapter_name] = adapter_path

        # ID management for tracking adapter versions
        self.adapter_train_ids = {
            adapter_name: self.short_id_generator()
            for adapter_name in self.adapter_names
        }

        # 
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

        # ---------------------------------------------------------
        # Init HF model, peft adapters
        # ---------------------------------------------------------
        self.hf_model = AutoModelForCausalLM.from_pretrained(
                    pretrained_model_name_or_path=self.model_name,
                    quantization_config=self.hf_quantization_config,
                    **self.hf_model_init_kwargs,
                )
        self.hf_adapters = {}
        self.optimizers = {}
        for ad_name in self.adapter_names:
            lora_config = LoraConfig(
                **self.lora_kwargs[adapter_name])
            hf_adapter = get_peft_model(
                    model=self.hf_model,
                    peft_config=lora_config,
                    adapter_name=adapter_name,
                )
            self.hf_adapters[ad_name] = hf_adapter
        

        # Init vLLM model
        self.vllm_model = vllm.LLM(
                    self.model_name,
                    **vllm_params,
                )


        self.hf_model = None
        self.adapter_wrappers = {}
        self.vllm_model = None
        
    def toggle_training_mode(self):
        for adn in self.adapter_names:
            self.adapter_train_ids[adn] = self.short_id_generator()
        self.vllm_model.sleep()

    def toggle_eval_mode(self):
        self.vllm_model.wake()

    def get_adapter_pointers(self) -> dict:
        pointers = {
            an : self.self.hf_adapters[an] for an in self.adapter_names
        }
        return pointers

    def prepare_adapter_eval(self, adapter_name: str):
        """
        
        """
        adapter_path = self.adapter_paths[adapter_name]
        self.current_lora_request = LoRARequest(
            adapter_name, 
            self.adapter_train_ids[adapter_name], 
            adapter_path
        )
        return 


    def prompt(self, contexts) -> list:
        """
        """

        # Apply chat template to contexts
        texts = self.tokenizer.apply_chat_template(
            contexts, 
            tokenize=False, 
            add_generation_prompt=True
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

    def export_adapter(
                self, 
                adapter_name: str, 
                export_path: str) -> None:
        
        """

        """

        

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
