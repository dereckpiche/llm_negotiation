"""
TODO: Figure out how to tweak SGlang not to go OOM when batch size is 32. See https://github.com/sgl-project/sglang/issues/6309.
"""


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
import httpx
import torch.nn as nn


logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))

AdapterID = str
PolicyID = str





class LeanLocalLLM:
    """
    TOWRITE
    """

    def __init__(
        self,
        name: str = "llama",
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        device: str = "cuda",
        shared_hf_llm_init_kwargs: dict = {},
        max_model_length: int = 8000,
        max_new_tokens: int = 128, # SGL Default: 128
        min_new_tokens: int = 1,
        stop_tokens_id: None | list[int] = None,
        temperature: float = 1.0,
        top_p: float = 1.0, # Top-p selects tokens from the smallest sorted set whose cumulative probability exceeds top_p. When top_p = 1, this reduces to unrestricted sampling from all tokens.
        top_k: int = -1, # Top-k randomly selects from the k highest-probability tokens. -1 means it is not used.
        frequency_penalty: float = 0.0,
        adapter_configs: dict = {},
        restrict_tokens=None,
        output_directory : str = "./models/",
        abort_sglang=False
    ) -> None:
        """
        Initializes the LocalLLM.
        """
        super().__init__()
        self.output_directory = output_directory
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
        self.sglang_adapter_ids = {
            adapter_id: adapter_id
            for adapter_id in self.adapter_ids
        }
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # Setup padding token to be same as EOS token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.needs_loading : dict[AdapterID, bool] = {adapter_id : False for adapter_id in self.adapter_ids}
        self.current_lora_request = None
        self.currently_loaded_adapter_id = None


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
        # Init Fast Inference Engine
        # ---------------------------------------------------------
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.min_new_tokens = min_new_tokens
        self.frequency_penalty = frequency_penalty
        self.init_sg_lang_server()


    def init_sg_lang_server(self) -> None:
        """
        TOWRITE
        """
        from transformers.utils import cached_file
        local_llm_path = os.path.split(cached_file(self.model_name, "config.json"))[0]
        # SGLang requires to load with LoRA to infer space required
        dummy_lora_path = os.path.join(self.save_path, self.adapter_ids[0])
        lora_str = "--lora-paths " + " ".join([str(lora_id)+"="+str(lora_path) for lora_id, lora_path in self.adapter_paths.items()])
        self.sglang_server_process, self.sglang_port = launch_server_cmd(
            f"""
            python3 -m sglang.launch_server --model-path {local_llm_path} \
            --host 0.0.0.0 \
            {lora_str} \
            --disable-radix-cache \
            """
        )
        # TODO: With the current SGL implementation, we cannot use radix caching with multiple LoRA adapters. Radix caching is great for our use case. We should check frequently if this has been enabled.
        print(f"LoRA String: {lora_str}")
        print(f"Local LLM Path: {local_llm_path}")
        self.sglang_sampling_params = {
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "max_new_tokens": self.max_new_tokens,
            "min_new_tokens": self.min_new_tokens,
            "frequency_penalty": self.frequency_penalty,
        }
        wait_for_server(f"http://localhost:{self.sglang_port}")
        self.gen_url     = f"http://localhost:{self.sglang_port}/generate"
        self.release_url = f"http://localhost:{self.sglang_port}/release_memory_occupation"
        self.resume_url  = f"http://localhost:{self.sglang_port}/resume_memory_occupation"
        self.load_weights_url = f"http://localhost:{self.sglang_port}/resume_memory_occupation"
        self.load_lora_url   = f"http://localhost:{self.sglang_port}/load_lora_adapter"
        self.unload_lora_url = f"http://localhost:{self.sglang_port}/unload_lora_adapter"


    def toggle_training_mode(self) -> None:
        for adn in self.adapter_ids:
            self.adapter_train_ids[adn] = self.short_id_generator()
        # remove allocated kv cache space from GPU
        requests.post(self.release_url, json={"tags": ["kv_cache"]}).raise_for_status()

    def toggle_eval_mode(self) -> None:
        # allocate kv cache space on GPU
        # TODO: make sure this is not allocated twice!
        requests.post(self.resume_url, json={"tags": ["kv_cache"]}).raise_for_status()


    def add_random_noise_to_current_adapter(self) -> None:
        # Add random noise to current adapter for debugging.
        pass

    def prepare_adapter_for_inference(self, adapter_id: AdapterID) -> None:
        """
        Ensure correct adapter is loaded in SGLang before generation.
        TODO: make efficient by keeping tabs of whether we made a new export /
        whether we need to load adapters each time!
        """
        sg_lang_id = self.sglang_adapter_ids[adapter_id]
        if self.needs_loading[adapter_id]:
            adapter_path = os.path.join(self.save_path, adapter_id)
            if os.path.exists(adapter_path):
                payload = {"lora_name": str(sg_lang_id)}
                requests.post(self.unload_lora_url, json=payload).raise_for_status()
                new_sglang_id = self.short_id_generator()
                logger.info(f"Loading adapter {adapter_id} from {adapter_path}. Previous SGLang id: {self.sglang_adapter_ids[adapter_id]}. New: {new_sglang_id}.")
                self.sglang_adapter_ids[adapter_id] = new_sglang_id
                payload = {
                "lora_name": str(new_sglang_id),
                "lora_path": str(adapter_path)
                }
                logger.info(f"Loaded adapter from {adapter_path}.")
                requests.post(self.load_lora_url, json=payload).raise_for_status()
                self.needs_loading[adapter_id] = False
        self.currently_loaded_adapter_id = adapter_id


    def get_training_policies(self) -> dict[PolicyID, nn.Module]:
        """
        Returns wrappers over the adapters which allows them be
        interfaced like regular PyTorch models.
        # TODO: create the adapter wrappers here
        See adapter_wrapper.py
        """
        trainable_objects = {
            an : self.hf_adapters[an] for an in self.adapter_ids
        }
        return trainable_objects

    def get_inference_policies(self) -> dict[PolicyID, Callable]:
        """
        TOWRITE
        """
        policies = {}
        for adapter_id in self.adapter_ids:
            # define policy func
            async def policy(prompt:list[dict], regex:str|None=None, _adapter_id=adapter_id):
                self.prepare_adapter_for_inference(adapter_id=_adapter_id)
                response = await self.generate(prompt, regex)
                return response
            policies[self.name+"/"+adapter_id] = policy
        return policies

    async def generate(self, prompt : list[dict], regex:str|None=None) -> str:
        """
        TODO : add json regex parser option
        """
        # Apply chat template to prompt
        prompt = self.tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True
        )
        sg_lang_adapter_id = self.sglang_adapter_ids[self.currently_loaded_adapter_id]
        print(f"Generating with adapter {self.currently_loaded_adapter_id}")
        self.sglang_sampling_params["regex"] = regex
        payload = {
            "text": [prompt],
            "lora_path": [sg_lang_adapter_id],
            "sampling_params": self.sglang_sampling_params
        }

        httpx_timeout = httpx.Timeout(
            connect=3600.0, read=3600.0, write=3600.0, pool=3600.0
        )
        async with httpx.AsyncClient(timeout=httpx_timeout) as client:
            resp = await client.post(self.gen_url, json=payload)
        response = resp.json()
        return response[0]['text']

    def export_adapters(self) -> None:
        """
        Any peft wrapper, by default, saves all adapters, not just the one currently loaded.
        """

        # New version of the adapters available
        for adapter_id in self.adapter_ids:
            self.needs_loading[adapter_id] = True

        # import random
        # self.save_path = self.save_path + str(random.randint(1,500))
        # print(f"Save path: {self.save_path}")
        # self.adapter_paths = {adapter_id:os.path.join(self.save_path, adapter_id) for adapter_id in self.adapter_ids}


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
        for adapter_id in self.adapter_ids:
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

    def short_id_generator(self) -> str:
        """
        Generates a short unique ID for tracking adapter versions.

        Returns:
            int: An 8-digit integer ID.
        """
        return str(uuid.uuid4().int)[:8]
