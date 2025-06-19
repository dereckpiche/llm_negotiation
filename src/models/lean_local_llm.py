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
from models.adapter_wrapper import AdapterWrapper
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
        shared_hf_llm_init_kwargs: dict = {},
        max_model_length: int = 8000,
        generation_args={
            "max_new_tokens": 300,
            "do_sample": True,
            "temperature": 1.0,
            "top_k": 0,
            "top_p": 1.0,
            "repetition_penalty": 1.0,
        },
        vllm_params: dict = {},
        adapter_configs: list[dict] =[{}],
        restrict_tokens=None,
        output_directory=None,
        abort_vllm=False
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
        self.shared_hf_llm_init_kwargs = shared_hf_llm_init_kwargs
        self.max_model_length = max_model_length
        self.restrict_tokens = restrict_tokens
        self.adapter_configs = adapter_configs
        self.adapter_ids = list(adapter_configs.keys())

        # TODO: load from external if exists! 

        # Path management / imports
        self.save_path = str(os.path.join(
            output_directory, 
            model_name, 
            "adapters"
            )
        )
        # ID management for tracking adapter versions
        self.adapter_train_ids = {
            adapter_id: self.short_id_generator()
            for adapter_id in self.adapter_ids
        }

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
        self.shared_hf_llm = AutoModelForCausalLM.from_pretrained(
                    pretrained_model_name_or_path=model_name,
                    **self.shared_hf_llm_init_kwargs,
                )
        self.hf_adapters = {}
        self.optimizers = {}
        for adapter_id in self.adapter_ids:
            hf_adapter = AdapterWrapper(
                shared_llm=self.shared_hf_llm,
                adapter_id=adapter_id,
                lora_config=adapter_configs[adapter_id],
                path=os.path.join(self.save_path, adapter_id)
            ).to(device)
            self.hf_adapters[adapter_id] = hf_adapter
        
        # ---------------------------------------------------------
        # Init vLLM stuff for fast inference
        # ---------------------------------------------------------
        from transformers.utils import cached_file
        local_llm_path = os.path.split(cached_file(model_name, "config.json"))[0]
        if not abort_vllm:
            self.vllm_model = vllm.LLM(model=local_llm_path, **vllm_params)
        else:
            self.vllm_model = None
        self.current_lora_request = None

        
    def toggle_training_mode(self) -> None:
        for adn in self.adapter_ids:
            self.adapter_train_ids[adn] = self.short_id_generator()
        if self.vllm_model is not None: self.vllm_model.sleep()

    def toggle_eval_mode(self) -> None:
        if self.vllm_model is not None: self.vllm_model.wake_up()

        

    def get_adapter_pointers(self) -> dict:
        pointers = {
            an : self.hf_adapters[an] for an in self.adapter_ids
        }
        return pointers

    def prepare_adapter_eval(self, adapter_id: str) -> None:
        """
        
        """
        adapter_path = os.path.join(self.save_path, adapter_id)
        if os.path.exists(adapter_path):
            self.current_lora_request = LoRARequest(
                adapter_id, 
                self.adapter_train_ids[adapter_id], 
                adapter_path
            )
        else: 
            self.current_lora_request = None


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

        # Restrict tokens of model # TODO, put in init??
        if self.restrict_tokens is not None:
            allowed_token_ids = []
            for token in self.restrict_tokens:
                token_ids = self.tokenizer(
                    token, 
                    add_special_tokens=False)["input_ids"]
                allowed_token_ids.append(token_ids[0])
            sampling_params = sampling_params.clone()
            sampling_params.allowed_token_ids = allowed_token_ids

        # Disable deterministic seed for more diversity
        sampling_params.seed = None

        if self.current_lora_request is not None:
            model_logger.info(
                f"Generating using vLLM with LoRA. "
                f"Current LoRA request ID is {self.current_lora_request.adapter_id}. "
                f"Current LoRA adapter path is {self.current_lora_request.path}."
            )
        else:
            model_logger.info(
                f"Generating using vLLM without LoRA. "
            )

        decoded = self.vllm_model.generate(
            prompts=texts,
            sampling_params=sampling_params,
            lora_request=self.current_lora_request,
        )
        responses = [d.outputs[0].text for d in decoded]
        return responses
    
    def export_adapters(self) -> None:
        """
        Any peft wrapper, by default, saves all adapters, not just self.
        """
        adapter_id = self.adapter_ids[0]
        self.hf_adapters[adapter_id].save_pretrained(self.save_path)

    def checkpoint_all_adapters(self, checkpoint_indicator: str) -> None:
        """
        Checkpoints all adapters to the configured output directory.
        """
        adapter_id = self.adapter_ids[0]
        output_dir = os.path.join(self.output_directory, "checkpoints")
        os.makedirs(output_dir, exist_ok=True)
        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        export_path = os.path.join(
            output_dir, f"{adapter_id}-{checkpoint_indicator}-{date_str}"
        )
        adapter_id = self.adapter_ids[0]
        self.hf_adapters[adapter_id].save_pretrained(export_path)


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
