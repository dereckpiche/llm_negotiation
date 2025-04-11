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
    A class to handle a single vLLM and HF instanciation pair and multiple adapters.

    args:
        name: str = "llama",
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        device: str = "cuda",
        hf_model_init_kwargs: dict = None,
        max_model_length: int = 8000,
    
    Adapter configs is a list of dictionaries. Each dictionary contains the following keys:
        - adapter_name: str
        - adapter_type: str (lora or full)
        - generation_args: dict 
        - lora_args: dict (Optional)
        - optimizer_method: str (Optional)
        - optimizer_kwargs: dict (Optional)
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
        optimizer_on_gpu_during_training=True,
        merge_weights_vllm_during_training=False,
        sleep_vllm_during_training=False,
        wake_vllm_during_eval=True,
        keep_vllm_during_training=False,
        keep_hf_during_training=True,
        keep_hf_during_eval=False,
        keep_vllm_during_eval=True,
        output_directory=None,
        base_seed: int = 42,
        vllm_params={},
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
        self.tokenizer = AutoTokenizer.from_pretrained(
            hf_model_init_kwargs["pretrained_model_name_or_path"]
        )

        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.bits_and_bytes_configs = (
            BitsAndBytesConfig(**bits_and_bytes_args) if bits_and_bytes_args else None
        )
        self.hf_sampling_params = generation_args
        self.vllm_sampling_params = SamplingParams(
            temperature=generation_args["temperature"],
            top_k=-1 if generation_args["top_k"] == 0.0 else generation_args["top_k"],
            top_p=generation_args["top_p"],
            max_tokens=generation_args["max_new_tokens"],
            repetition_penalty=generation_args["repetition_penalty"],
        )

        self.optimizer_on_gpu_during_training = optimizer_on_gpu_during_training
        self.merge_weights_vllm_before_eval = merge_weights_vllm_during_training
        self.sleep_vllm_during_training = sleep_vllm_during_training
        self.wake_vllm_during_eval = wake_vllm_during_eval
        self.keep_vllm_during_training = keep_vllm_during_training
        self.keep_hf_during_training = keep_hf_during_training
        self.keep_hf_during_eval = keep_hf_during_eval
        self.keep_vllm_during_eval = keep_vllm_during_eval
        self.train_with = train_with
        self.eval_with = eval_with

        self.hf_model = None
        self.vllm_model = None

        self.current_optimizer = None
        self.current_lora_request = None
        self.adapter_configs = adapter_configs
        self.adapter_names = self.adapter_configs.keys()
        self.active_adapters = {adapter_name: False for adapter_name in self.adapter_names}
        self.adapter_types = {item[0]: item[1]["type"] for item in self.adapter_configs.items()}
        self.current_adapter_name = None
        self.optimizer_paths = {adapter_name: os.path.join(self.output_directory, adapter_name, "optimizer.pt") for adapter_name in self.adapter_names}
        self.optimizer_methods = {item[0]: item[1]["optimizer_method"] for item in self.adapter_configs.items()}
        self.optimizer_kwargs = {item[0]: item[1]["optimizer_kwargs"] for item in self.adapter_configs.items()}
        self.lora_kwargs = {item[0]: item[1]["lora_kwargs"] for item in self.adapter_configs.items()}


        self.base_seed = base_seed
        self.at_least_one_full_adapter = any(config["type"] == "full" for config in self.adapter_configs.values())
        self.adapter_paths = {
            adapter_name: os.path.join(self.output_directory, adapter_name)
            for adapter_name in self.adapter_names
        }
        self.adapter_train_ids = {  
            adapter_name: self.short_id_generator() for adapter_name in self.adapter_names
        }
        self.adapter_eval_ids = deepcopy(self.adapter_train_ids)

        self.current_vllm_name = None
        self.vllm_ids = deepcopy(self.adapter_train_ids)
 
     
    def prepare_adapter_train(self, adapter_name: str):
        """
        Prepares the agent for training with the specified adapter.
        """

        if not self.keep_vllm_during_training:
            self.vllm_model = None
            import gc
            gc.collect()
            torch.cuda.empty_cache()

        if self.sleep_vllm_during_training:
            self.vllm_model.sleep()

        # Creating new model version, must change training id.
        self.adapter_train_ids[adapter_name] = self.short_id_generator()

        self.log_gpu_usage(
            f"Before loading HF model with adapter {adapter_name} for training."
        )

        model_logger.info(f"Preparing adapter {adapter_name} for training.")

        start_time = time.time()

        self.current_adapter_name = adapter_name
        adapter_path = self.adapter_paths[self.current_adapter_name]
        adapter_type = self.adapter_types[self.current_adapter_name]

        if self.hf_model is None:
            if os.path.exists(adapter_path) and adapter_name == "full" :
                self.hf_model = AutoModelForCausalLM.from_pretrained(
                    adapter_path,
                    **self.hf_model_init_kwargs,
                )
            else: 
                self.hf_model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    **self.hf_model_init_kwargs,
                )
        if adapter_type == "full":
            model_logger.info("Setting up full model training (no LoRA)")
            for param in self.hf_model.parameters():
                param.requires_grad = True

        elif adapter_type == "lora":
            if adapter_name in self.hf_model.list_adapters():
                self.hf_model.set_adapter(adapter_name)
            elif os.path.exists(adapter_path):
                self.hf_model.load_adapter(adapter_path)
            else:
                self.hf_model.add_adapter(adapter_name, lora_config=LoraConfig(**self.lora_kwargs[adapter_name]))
            
        # Load the optimizer
        self.current_optimizer = self.optimizer_methods[adapter_name](
            self.hf_model.parameters(),
            **self.optimizer_kwargs[adapter_name]
        )
        if os.path.exists(self.optimizer_paths[adapter_name]):
            self.current_optimizer.load_state_dict(torch.load(self.optimizer_paths[adapter_name]))

        if self.optimizer_on_gpu_during_training:
            self.current_optimizer.to("cuda")
        
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
        adapter_type = self.adapter_types[adapter_name]
        adapter_path = self.adapter_paths[adapter_name]
        self.current_adapter_name = adapter_name
        is_adapter_changed = self.current_adapter_name != self.current_vllm_name
        is_adapter_updated = self.vllm_ids[self.current_adapter_name] != self.adapter_train_ids[self.current_adapter_name]
        is_full_adapter = adapter_type == "full"
        weight_merge_required = is_adapter_changed and is_adapter_updated and is_full_adapter

        if not self.keep_hf_during_eval:
            model_logger.info(f"Deleting HF model for evaluation with {adapter_name}.")
            del self.hf_model
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            self.hf_model = None

        if self.eval_with == "vllm":

            if self.vllm_model is None:

                self.log_gpu_usage(f"Before loading VLLM model with {adapter_name}.")
                start_time = time.time()

                if self.at_least_one_full_adapter:
                    import os
                    os.environ["VLLM_USE_V1"] = '0'
                    self.vllm_model = LLM(
                        self.model_name,
                        **self.vllm_params,
                        worker_extension_cls='models.updatable_worker.UpdatableWorkerExtension'
                    )
                else:
                    self.vllm_model = LLM(
                        self.model_name,
                        **self.vllm_params
                    )
                end_time = time.time()
                compute_logger.info(f"VLLM model loading time: {end_time - start_time:.2f} seconds.")
                self.log_gpu_usage(f"After loading VLLM model with {adapter_name}.")

            # Update vllm name and id
            if is_adapter_changed:
                self.current_vllm_name = adapter_name
            if is_adapter_updated:
                self.vllm_ids[adapter_name] = self.adapter_train_ids[adapter_name]

            # Full weight switch    
            if weight_merge_required:
                model_logger.info(f"Merging weights for {adapter_name} from {adapter_path} to vLLM model.")
                assert self.hf_model is not None, "HF model is not loaded. Cannot merge weights."
                for name, param in self.hf_model.named_parameters():
                    self.vllm_model.collective_rpc('update_weight', args=(name, param.data))

            # LoRA adapting 
            if adapter_type == "lora":
                self.current_lora_request = LoRARequest(
                    adapter_name, self.adapter_train_ids[adapter_name], adapter_path
                )

            else:
                self.current_lora_request = None

            if self.wake_vllm_during_eval:
                self.vllm_model.wake_up()

        self.current_adapter_name = adapter_name

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

            if self.current_lora_request is not None:
                model_logger.info(
                    f"Generating using VLLM with LoRA. Current LoRA request ID is {self.current_lora_request.adapter_id}. Current LoRA adapter path is {self.current_lora_request.path}."
                )
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
        adapter_type = self.adapter_types[self.current_adapter_name]
        os.makedirs(adapter_path, exist_ok=True)
        
        if adapter_type == "full":
            model_logger.info(f"Saving full model to {adapter_path}")
        elif adapter_type == "lora":
            model_logger.info(f"Saving LoRA weights to {adapter_path}")

        self.hf_model.save_pretrained(adapter_path)
            
        # Save optimizer state
        optimizer_state_path = os.path.join(adapter_path, "optimizer.pt")
        torch.save(self.current_optimizer.state_dict(), optimizer_state_path)
        model_logger.info(f"Optimizer state saved to {optimizer_state_path}")

        # For vllm
        # TODO (Muqeeth): check with Dereck if this is needed.
        # with open(os.path.join(adapter_path, "config.json"), "w") as f:
        #     json.dump({"model_type": "llama"}, f)

    def short_id_generator(self):
        """
        Generates a unique random short 8-digit number.
        """
        return str(uuid.uuid4().int)[:8]

        
