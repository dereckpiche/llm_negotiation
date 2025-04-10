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

from utils.common_imports import *

compute_logger = logging.getLogger("compute_logger")
memory_logger = logging.getLogger("memory_logger")
model_logger = logging.getLogger("model_logger")


class LocalLLM:
    """
    LocalLLM is an agent that utilizes HuggingFace models for causal language modeling.
    It supports training using Proximal Policy Optimization (PPO) and saving/loading models.
    """

    def __init__(
        self,
        name: str = "llama",
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        device: str = "cuda",
        pretrained_args={
            "pretrained_model_name_or_path": "meta-llama/Llama-3.1-8B-Instruct",
            "torch_dtype": "bfloat16",
            "device_map": "auto",
            "attn_implementation": "flash_attention_2",
        },
        max_model_length=8000,
        bits_and_bytes_args=None,
        lora_args={
            "task_type": "TaskType.CAUSAL_LM",
            "r": 64,
            "lora_alpha": 32,
            "lora_dropout": 0.0,
            "target_modules": "all-linear",
        },
        adapter_names=["ad_alice", "ad_bob"],
        generation_args={
            "max_new_tokens": 300,
            "do_sample": True,
            "temperature": 1.0,
            "top_k": 0,
            "top_p": 1.0,
            "repetition_penalty": 1.0,
        },
        include_value_head=False,
        keep_vllm_during_training=False,
        keep_hf_during_training=True,
        keep_hf_during_eval=False,
        keep_vllm_during_eval=True,
        eval_with="vllm",
        train_with="hf",
        output_directory=None,
        base_seed: int = 42,
        vllm_params={},
        optimizer_method="AdamW",
        optimizer_kwargs={"lr": 1e-6, "weight_decay": 0.0},
    ) -> None:
        """
        Initializes the LocalLLM.
        """
        super().__init__()
        self.vllm_params = vllm_params
        self.name = name
        self.device = torch.device(device) if device else torch.device("cuda")
        self.model_name = model_name
        self.include_value_head = include_value_head
        self.pretrained_args = pretrained_args
        self.max_model_length = max_model_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_args["pretrained_model_name_or_path"]
        )

        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.bits_and_bytes_configs = (
            BitsAndBytesConfig(**bits_and_bytes_args) if bits_and_bytes_args else None
        )
        self.lora_args = lora_args
        self.hf_sampling_params = generation_args
        self.vllm_sampling_params = SamplingParams(
            temperature=generation_args["temperature"],
            top_k=-1 if generation_args["top_k"] == 0.0 else generation_args["top_k"],
            top_p=generation_args["top_p"],
            max_tokens=generation_args["max_new_tokens"],
            repetition_penalty=generation_args["repetition_penalty"],
        )
        self.lora_config = LoraConfig(**lora_args)
        self.active_adapters = {adapter_name: False for adapter_name in adapter_names}
        self.current_adapter_name = None
        self.hf_model = None
        self.vllm_model = None
        self.keep_vllm_during_training = keep_vllm_during_training
        self.keep_hf_during_training = keep_hf_during_training
        self.keep_hf_during_eval = keep_hf_during_eval
        self.keep_vllm_during_eval = keep_vllm_during_eval
        self.train_with = train_with
        self.eval_with = eval_with
        self.output_directory = output_directory
        self.adapters_active = False
        self.vllm_id = 0
        self.hf_id = 0
        self.adapter_paths = {
            adapter_name: os.path.join(self.output_directory, adapter_name)
            for adapter_name in adapter_names
        }
        self.adapter_train_ids = {
            adapter_name: uuid.uuid4() for adapter_name in adapter_names
        }

        self.lora_request = None

        # set random seeds
        self.base_seed = base_seed
        self.adapter_names = adapter_names
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            **self.pretrained_args, quantization_config=self.bits_and_bytes_configs
        )
        for adapter_name in adapter_names:
            adapter_path = self.adapter_paths[adapter_name]
            if os.path.exists(adapter_path):
                model_logger.info(f"Loading adapter {adapter_name} from {adapter_path}")
                self.hf_model.load_adapter(adapter_path, adapter_name)
            else:
                self.hf_model.add_adapter(self.lora_config, adapter_name)
        self.optimizer = None
        self.adapter_optimizers = {}
        for adapter_name in adapter_names:
            self.hf_model.set_adapter(adapter_name)
            # set_adapters has the correct trainable parameters.
            self.adapter_optimizers[adapter_name] = getattr(
                torch.optim, optimizer_method
            )(self.hf_model.parameters(), **optimizer_kwargs)
            optimizer_state_path = os.path.join(
                self.adapter_paths[adapter_name], "optimizer.pt"
            )
            if os.path.exists(optimizer_state_path):
                model_logger.info(
                    f"Loading optimizer state from {optimizer_state_path}"
                )
                self.adapter_optimizers[adapter_name].load_state_dict(
                    torch.load(optimizer_state_path)
                )

    def prepare_adapter_train(self, adapter_name: str):
        """
        Prepares the agent for training with the specified adapter.
        """
        if not self.keep_vllm_during_training:
            self.vllm_model.sleep()

        # Creating new model version, must change training id.
        self.adapter_train_ids[adapter_name] = uuid.uuid4()

        self.log_gpu_usage(
            f"Before loading HF model with adapter {adapter_name} for training."
        )

        model_logger.info(f"Preparing adapter {adapter_name} for training.")

        start_time = time.time()

        self.current_adapter_name = adapter_name
        adapter_path = self.adapter_paths[self.current_adapter_name]
        if self.train_with == "hf":
            self.hf_model.set_adapter(adapter_name)
            # set the right optimizer for adapter_name
            self.optimizer = self.adapter_optimizers[adapter_name]
            # Log trainable parameters
            total_params = sum(p.numel() for p in self.hf_model.parameters())
            trainable_params = sum(
                p.numel() for p in self.hf_model.parameters() if p.requires_grad
            )
            model_logger.info(f"Total Parameters: {total_params}")
            model_logger.info(
                f"Trainable Parameters: {trainable_params} ({trainable_params/total_params:.2%})"
            )

        end_time = time.time()
        compute_logger.info(
            f"HF model loading time: {end_time - start_time:.2f} seconds."
        )

        self.log_gpu_usage(
            f"After loading HF model with adapter {adapter_name} for training."
        )

    def prepare_adapter_eval(self, adapter_name: str, seed_offset: int = 0):
        """
        Prepares the agent for evaluation with the specified adapter.
        """
        model_logger.info(f"Preparing adapter {adapter_name} for evaluation.")

        uuid_obj = self.adapter_train_ids[adapter_name]
        hash_object = hashlib.sha256(uuid_obj.bytes)
        hash_int = int(hash_object.hexdigest(), 16)
        id = int(hash_int % 10000000)
        adapter_path = self.adapter_paths[adapter_name]
        if os.path.exists(adapter_path):
            self.lora_request = LoRARequest(
                f"dond_lora_{adapter_name}", id, adapter_path
            )
        else:
            self.lora_request = None

        self.current_adapter_name = adapter_name

        if self.eval_with == "vllm":
            if self.vllm_model is None:
                self.log_gpu_usage(f"Before loading VLLM model with {adapter_name}.")

                start_time = time.time()
                # TODO (Muqeeth): check if its okay to have seed fixed since we update lora parameters anyway.
                self.vllm_model = LLM(
                    self.model_name,
                    enable_lora=True,
                    max_lora_rank=256,
                    seed=self.base_seed,
                    dtype=torch.bfloat16,
                    **self.vllm_params,
                )
                end_time = time.time()
                compute_logger.info(
                    f"VLLM model loading time: {end_time - start_time:.2f} seconds."
                )
                self.log_gpu_usage(f"After loading VLLM model with {adapter_name}.")
            else:
                self.vllm_model.wake_up()

    def log_gpu_usage(self, message: str) -> None:
        """
        Logs the GPU memory usage.

        Args:
            message (str): A message to include in the log.
        """
        free_memory, total_memory = torch.cuda.mem_get_info()
        gpu_memory = (total_memory - free_memory) / (1024**3)
        memory_logger.info(f"{message}: GPU memory allocated: {gpu_memory:.2f} GB")

    def prompt(self, contexts) -> str:
        """
        Generates a response from the model based on the provided contexts.

        Args:
            contexts (List[dict]): The contexts for generation.

        scores:
            str: The generated response from the model.
        """
        adapter_path = self.adapter_paths[self.current_adapter_name]
        # print(f"len of contexts and current adapter : {len(contexts), self.current_adapter_name}")
        # print(f"sample context : {contexts[0]}")
        if len(contexts) == 0:
            return []

        # TODO (Muqeeth): Vllm has issue with repeating bos_token twice (https://github.com/vllm-project/vllm/pull/15695/files)
        texts = self.tokenizer.apply_chat_template(
            contexts, tokenize=False, add_generation_prompt=True
        )

        start_time = time.time()

        if self.eval_with == "vllm":
            # Seeding vllm is causing it to be determinisitc acoross prompts, but if we seed earlier then we can replicate generations
            # self.vllm_sampling_params.seed = self.base_seed + seed_offset
            # print("Seed used for generation: ", self.vllm_sampling_params.seed)
            self.vllm_sampling_params.seed = None

            if self.lora_request is not None:
                model_logger.info(
                    f"Generating using VLLM with LoRA. Current LoRA request ID is {self.lora_request.adapter_id}. Current LoRA adapter path is {self.lora_request.path}."
                )
                decoded = self.vllm_model.generate(
                    texts,
                    sampling_params=self.vllm_sampling_params,
                    lora_request=self.lora_request,
                )
            else:
                model_logger.info("Generating using VLLM without LoRA.")
                decoded = self.vllm_model.generate(
                    texts, sampling_params=self.vllm_sampling_params
                )
            responses = [d.outputs[0].text for d in decoded]
            del decoded

        else:
            model_logger.error(f"Unsupported generation method: {self.eval_with}")
            return []

        end_time = time.time()
        # compute_logger.info(
        #     f"Generation completed in {end_time - start_time:.2f} seconds using {self.eval_with}."
        # )

        return responses

    def export_current_adapter(self) -> None:
        """
        Saves only the LoRA weights to a specified directory. If the directory
        already exists, it deletes the existing directory before saving.
        """
        adapter_path = self.adapter_paths[self.current_adapter_name]

        # if os.path.exists(adapter_path):
        #     shutil.rmtree(adapter_path)
        #     logging.info(f"Existing directory '{adapter_path}' deleted.")

        os.makedirs(adapter_path, exist_ok=True)
        # Save only the LoRA weights
        self.hf_model.save_pretrained(adapter_path)
        model_logger.info(f"LoRA weights saved to {adapter_path}")
        # Save optimizer state
        optimizer_state_path = os.path.join(adapter_path, "optimizer.pt")
        torch.save(self.optimizer.state_dict(), optimizer_state_path)
        model_logger.info(f"Optimizer state saved to {optimizer_state_path}")

        # For vllm
        # TODO (Muqeeth): check with Dereck if this is needed.
        with open(os.path.join(adapter_path, "config.json"), "w") as f:
            json.dump({"model_type": "llama"}, f)
