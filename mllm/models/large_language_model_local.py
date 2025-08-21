"""
TODO: Figure out how to tweak SGlang not to go OOM when batch size is 32. See https://github.com/sgl-project/sglang/issues/6309.
"""

import logging
import os
import sys
import uuid
from collections.abc import Callable
from copy import deepcopy
from datetime import datetime
from typing import Literal

import httpx
import requests
import torch
import torch.nn as nn
from sglang.utils import (
    launch_server_cmd,
    print_highlight,
    terminate_process,
    wait_for_server,
)
from torch.optim import SGD, Adam, AdamW, RMSprop
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead

from mllm.models.adapter_training_wrapper import AdapterWrapper
from mllm.models.inference_backend_dummy import DummyInferenceBackend
from mllm.models.inference_backend_sglang import SGLangOfflineBackend
from mllm.models.inference_backend_vllm import VLLMAsyncBackend

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
        llm_id: str = "base_llm",
        model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
        device: str = "cuda",
        hf_kwargs: dict = {},
        adapter_configs: dict = {},
        output_directory: str = "./models/",
        inference_backend: Literal["vllm", "sglang", "dummy"] = "vllm",
        inference_backend_sampling_params: dict = {},
        inference_backend_init_kwargs: dict = {},
        initial_adapter_paths: dict[str, str] | None = None,
    ):
        self.inference_backend_name = inference_backend
        self.output_directory = output_directory
        self.llm_id = llm_id
        self.device = torch.device(device) if device else torch.device("cuda")
        self.model_name = model_name
        self.adapter_configs = adapter_configs
        self.adapter_ids = list(adapter_configs.keys())

        # Optional user-specified initial adapter weight locations (local or HF Hub)
        # Format: {adapter_id: path_or_repo_id}
        self.initial_adapter_paths: dict[str, str] | None = initial_adapter_paths

        # Path management / imports
        self.save_path = str(os.path.join(output_directory, model_name, "adapters"))
        self.adapter_paths = {
            adapter_id: os.path.join(self.save_path, adapter_id)
            for adapter_id in self.adapter_ids
        }
        # ID management for tracking adapter versions
        self.adapter_train_ids = {
            adapter_id: self.short_id_generator() for adapter_id in self.adapter_ids
        }
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # Setup padding token to be same as EOS token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.weights_got_updated: dict[AdapterID, bool] = {
            adapter_id: False for adapter_id in self.adapter_ids
        }
        self.current_lora_request = None
        self.currently_loaded_adapter_id = None

        # ---------------------------------------------------------
        # Init HF model, peft adapters
        # ---------------------------------------------------------
        self.shared_hf_llm = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name,
            **hf_kwargs,
        )
        self.hf_adapters = {}
        self.optimizers = {}
        for adapter_id in self.adapter_ids:
            # Prefer output-folder path if it exists; else fall back to user-specified initial path if provided
            output_path = os.path.join(self.save_path, adapter_id)
            chosen_path: str | None = None
            if os.path.isdir(output_path) and os.listdir(output_path):
                chosen_path = output_path
                logger.info(
                    f"Initializing adapter '{adapter_id}': using existing weights from output folder '{chosen_path}'."
                )
            elif (
                self.initial_adapter_paths and adapter_id in self.initial_adapter_paths
            ):
                chosen_path = self.initial_adapter_paths[adapter_id]
                logger.info(
                    f"Initializing adapter '{adapter_id}': using provided initial path '{chosen_path}'."
                )
            else:
                logger.info(
                    f"Initializing adapter '{adapter_id}': no initial weights provided or found; starting from scratch."
                )

            hf_adapter = AdapterWrapper(
                shared_llm=self.shared_hf_llm,
                adapter_id=adapter_id,
                lora_config=adapter_configs[adapter_id],
                path=chosen_path,
            ).to(device)
            self.hf_adapters[adapter_id] = hf_adapter
        # Persist current state of all adapters (ensures remote loads are cached to disk)
        self.export_adapters()

        # ---------------------------------------------------------
        # Init inference inference_backend
        # ---------------------------------------------------------

        if inference_backend == "sglang":
            self.inference_backend = SGLangOfflineBackend(
                model_name=self.model_name,
                save_path=self.save_path,
                adapter_paths=self.adapter_paths,
                tokenizer=self.tokenizer,
                kwargs=inference_backend_init_kwargs,
            )
        elif inference_backend == "vllm":
            self.inference_backend = VLLMAsyncBackend(
                model_name=self.model_name,
                adapter_paths=self.adapter_paths,
                tokenizer=self.tokenizer,
                engine_init_kwargs=inference_backend_init_kwargs,
                sampling_params=inference_backend_sampling_params,
            )
        elif inference_backend == "dummy":
            self.inference_backend = DummyInferenceBackend()
        else:
            raise ValueError(f"Unknown inference_backend: {inference_backend}")

    def get_inference_policies(self) -> dict[PolicyID, Callable]:
        """
        TOWRITE
        """
        policies = {}
        for adapter_id in self.adapter_ids:
            # define policy func
            async def policy(
                prompt: list[dict], regex: str | None = None, _adapter_id=adapter_id
            ):
                self.prepare_adapter_for_inference(adapter_id=_adapter_id)
                response = await self.generate(prompt, regex)
                # response = await self.generate(prompt, "^<(A|B)>$")
                return response

            policies[self.llm_id + "/" + adapter_id] = policy
        return policies

    def get_adapter_modules(self) -> dict[PolicyID, nn.Module]:
        """
        Returns wrappers over the adapters which allows them be
        interfaced like regular PyTorch models.
        # TODO: create the adapter wrappers here
        See adapter_wrapper.py
        """
        trainable_objects = {an: self.hf_adapters[an] for an in self.adapter_ids}
        return trainable_objects

    async def toggle_training_mode(self) -> None:
        for adn in self.adapter_ids:
            self.adapter_train_ids[adn] = self.short_id_generator()
        self.inference_backend.toggle_training_mode()

    async def toggle_eval_mode(self) -> None:
        self.inference_backend.toggle_eval_mode()

    def prepare_adapter_for_inference(self, adapter_id: AdapterID) -> None:
        self.inference_backend.prepare_adapter(
            adapter_id, weights_got_updated=self.weights_got_updated[adapter_id]
        )
        self.currently_loaded_adapter_id = adapter_id
        self.weights_got_updated[adapter_id] = False

    async def generate(self, prompt: list[dict], regex: str | None = None) -> str:
        # Chat templating
        prompt_text = self.tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
        return await self.inference_backend.generate(
            prompt_text=prompt_text, regex=regex
        )

    def export_adapters(self) -> None:
        """
        Any peft wrapper, by default, saves all adapters, not just the one currently loaded.
        """

        # New version of the adapters available
        for adapter_id in self.adapter_ids:
            self.weights_got_updated[adapter_id] = True

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

    def short_id_generator(self) -> str:
        """
        Generates a short unique ID for tracking adapter versions.

        Returns:
            int: An 8-digit integer ID.
        """
        return str(uuid.uuid4().int)[:8]
