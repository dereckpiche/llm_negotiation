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
from mllm.models.adapter_training_wrapper import AdapterWrapper
from torch.optim import SGD, Adam, AdamW, RMSprop
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead
import subprocess, json, os, sys, time, requests
from sglang.utils import launch_server_cmd
from sglang.utils import wait_for_server, print_highlight, terminate_process
from collections.abc import Callable
from mllm.utils.common_imports import *

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
        sglang_params: dict = {},
        adapter_configs: dict = {},
        restrict_tokens=None,
        output_directory=None,
        abort_sglang=False
    ) -> None:
        """
        Initializes the LocalLLM.
        """
        super().__init__()
        self.output_directory = output_directory
        self.sglang_params = sglang_params
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
        self.adapter_paths = {adapter_id:os.path.join(self.save_path, adapter_id) for adapter_id in self.adapter_ids}
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
        # self.sglang_sampling_params = SamplingParams(
        #     temperature=generation_args["temperature"],
        #     top_k=-1 if generation_args["top_k"] == 0.0 else generation_args["top_k"],
        #     top_p=generation_args["top_p"],
        #     max_tokens=generation_args["max_new_tokens"],
        #     repetition_penalty=generation_args["repetition_penalty"],
        # )

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
        self.export_adapters()
        # ---------------------------------------------------------
        # Init sglang stuff for fast inference
        # ---------------------------------------------------------
        from transformers.utils import cached_file
        local_llm_path = os.path.split(cached_file(model_name, "config.json"))[0]
        # SGLang requires to load with LoRA to infer space required
        dummy_lora_path = os.path.join(self.save_path, self.adapter_ids[0])
        lora_str = "--lora-paths " + " ".join([str(lora_id)+"="+str(lora_path) for lora_id, lora_path in self.adapter_paths.items()])
        self.sglang_server_process, self.sglang_port = launch_server_cmd(
            "python3 -m sglang.launch_server " + \
            f"--model-path {local_llm_path} " + \
            "--host 0.0.0.0 " + \
            lora_str + \
            " --enable-memory-saver" + \
            " --disable-radix-cache"
        )
        wait_for_server(f"http://localhost:{self.sglang_port}")
        self.gen_url     = f"http://localhost:{self.sglang_port}/generate"
        self.release_url = f"http://localhost:{self.sglang_port}/release_memory_occupation"
        self.resume_url  = f"http://localhost:{self.sglang_port}/resume_memory_occupation"
        self.load_weights_url = f"http://localhost:{self.sglang_port}/resume_memory_occupation"
        self.load_lora_url   = f"http://localhost:{self.sglang_port}/load_lora_adapter"
        self.unload_lora_url = f"http://localhost:{self.sglang_port}/unload_lora_adapter"
        self.current_lora_request = None


    def toggle_training_mode(self) -> None:
        for adn in self.adapter_ids:
            self.adapter_train_ids[adn] = self.short_id_generator()
        # remove allocated kv cache space from GPU
        requests.post(self.release_url, json={"tags": ["kv_cache"]}).raise_for_status()

    def toggle_eval_mode(self) -> None:
        # allocate kv cache space on GPU
        # TODO: make sure this is not allocated twice!
        requests.post(self.resume_url, json={"tags": ["kv_cache"]}).raise_for_status()


    def set_adapter_eval(self, adapter_id: str) -> None:
        """
        Ensure correct adapter is loaded in SGLang before generation.
        TODO: make efficient by keeping tabs of whether we made a new export /
        whether we need to load adapters each time!
        """
        adapter_path = os.path.join(self.save_path, adapter_id)
        print(str(adapter_id))
        print(str(adapter_path))
        if os.path.exists(adapter_path):
            payload = {"lora_name": str(adapter_id)}
            requests.post(self.unload_lora_url, json=payload).raise_for_status()
            payload = {
               "lora_name": str(adapter_id),
               "lora_path": str(adapter_path)
            }
            requests.post(self.load_lora_url, json=payload).raise_for_status()

    def get_trainable_objects(self) -> dict:
        """
        TOWRITE
        """
        trainable_objects = {
            an : self.hf_adapters[an] for an in self.adapter_ids
        }
        return trainable_objects

    def get_callable_objects(self) -> dict[str, Callable]:
        """
        TOWRITE
        """
        policies = {}
        for adapter_id in self.adapter_ids:
            # define policy func
            def policy(prompt:list[dict]):
                self.set_adapter_eval(adapter_id=adapter_id)
                return self.generate(prompt)
            policies[self.name+"/"+adapter_id] = policy
        return policies

    def generate(self, prompt) -> str:
        """
        """
        # Apply chat template to prompt
        prompt = self.tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True
        )
        print(prompt)
        payload = {"text": prompt}
        response = requests.post(self.gen_url, json=payload).json()
        return response["text"]

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
