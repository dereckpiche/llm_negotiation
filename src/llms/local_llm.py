from utils.common_imports import *
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
import os
import shutil
from trl import (
    AutoModelForCausalLMWithValueHead,
)
from peft import PeftModel
import os
import logging
import time

from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import PeftModel
from peft import LoraConfig, get_peft_model
from trl import AutoModelForCausalLMWithValueHead
import torch
import gc
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import json
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

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
        name: str = "your_friendly_llm",
        model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        device: str = "cuda",
        pretrained_args=None,
        max_model_length=6000,
        bits_and_bytes_args=None,
        lora_args=None,
        adapter_names=None,
        generation_args=None,
        sgl_args=None,
        include_value_head=True,
        keep_vllm_during_training=False,
        keep_hf_during_training=True,
        keep_hf_during_eval=False,
        keep_vllm_during_eval=True,
        eval_with="vllm",
        train_with="hf",
        output_directory=None,
        random_seed: int = 42,
    ) -> None:
        """
        Initializes the LocalLLM.
        """
        super().__init__()
        self.name = name
        self.device = torch.device(device) if device else torch.device("cuda:0")
        self.model_name = model_name
        self.include_value_head = include_value_head
        self.pretrained_args = pretrained_args
        self.max_model_length = max_model_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_args["pretrained_model_name_or_path"]
        )

        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.bits_and_bytes_configs = BitsAndBytesConfig(**bits_and_bytes_args) if bits_and_bytes_args else None
        self.lora_args = lora_args
        self.hf_sampling_params = generation_args
        self.vllm_sampling_params = SamplingParams(
            temperature=generation_args["temperature"],
            top_k=-1 if generation_args["top_k"] == 0.0 else generation_args["top_k"],
            top_p=generation_args["top_p"],
            max_tokens=generation_args["max_new_tokens"],
        )
        self.sgl_sampling_params = sgl_args if sgl_args is not None else self.hf_sampling_params
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
        self.adapters = {
            adapter_name: os.path.join(self.output_directory, adapter_name) if os.path.isdir(os.path.join(output_directory, adapter_name)) else None
            for adapter_name in adapter_names
        }

        # set random seeds
        self.random_seed = random_seed

    def prepare_adapter_train(self, adapter_name: str):
        """
        Prepares the agent for training with the specified adapter.
        """
        ######################################
        # Prepare training model with HF
        ######################################

        torch.cuda.set_device(self.device.index)

        self.destroy_hf()
        if not self.keep_vllm_during_training:
            self.destroy_vllm()
        if not self.keep_hf_during_training:
            self.destroy_hf()

        self.log_gpu_usage(f"Before loading HF model with adapter {adapter_name} for training.")

        model_logger.info(f"Preparing adapter {adapter_name} for training.")

        start_time = time.time()


        self.current_adapter_name = adapter_name
        adapter_path = self.adapters[self.current_adapter_name]
        if self.train_with == "hf":
            if adapter_path is None:
                self.hf_model = AutoModelForCausalLM.from_pretrained(
                    **self.pretrained_args,
                    quantization_config=self.bits_and_bytes_configs
                )
                self.hf_model = get_peft_model(self.hf_model, self.lora_config)
                self.hf_model.train()
                model_logger.info(f"Adapter '{self.current_adapter_name}' added to HF.")
            else:
                self.hf_model = AutoModelForCausalLM.from_pretrained(
                    **self.pretrained_args,
                    quantization_config=self.bits_and_bytes_configs
                )
                self.hf_model = PeftModel.from_pretrained(
                    model=self.hf_model,
                    model_id=adapter_path,
                    is_trainable=True
                )
                self.hf_model.train()
                model_logger.info(f"Adapter '{self.current_adapter_name}' loaded to HF from {adapter_path}.")

            # Ensure HF model is moved to the proper device
            self.hf_model.to(self.device)

            # Log trainable parameters
            total_params = sum(p.numel() for p in self.hf_model.parameters())
            trainable_params = sum(p.numel() for p in self.hf_model.parameters() if p.requires_grad)
            model_logger.info(f"Total Parameters: {total_params}")
            model_logger.info(f"Trainable Parameters: {trainable_params} ({trainable_params/total_params:.2%})")


        end_time = time.time()
        compute_logger.info(f"HF model loading time: {end_time - start_time:.2f} seconds.")

        self.log_gpu_usage(f"After loading HF model with adapter {adapter_name} for training.")


    def prepare_adapter_eval(self, adapter_name: str):
        """
        Prepares the agent for evaluation with the specified adapter.
        """
        torch.cuda.set_device(self.device.index)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.device.index)
        if not self.keep_hf_during_eval:
            self.destroy_hf()
        if not self.keep_vllm_during_eval:
            self.destroy_vllm()
        model_logger.info(f"Preparing adapter {adapter_name} for evaluation.")
        self.current_adapter_name = adapter_name

        ######################################
        # Prepare evaluation model with vLLM
        ######################################
        if self.eval_with == "vllm":
            if self.vllm_model is None:
                self.log_gpu_usage(f"Before loading VLLM model with {adapter_name}.")
                start_time = time.time()
                self.vllm_model = LLM(self.model_name,
                                      enable_lora=True,
                                      max_lora_rank=256,
                                      seed=self.random_seed,
                                      max_model_len=self.max_model_length,
                                      tensor_parallel_size=1,
                                      dtype=self.pretrained_args["torch_dtype"],
                                      device=self.device
                                      )
                end_time = time.time()
                compute_logger.info(f"VLLM model loading time: {end_time - start_time:.2f} seconds.")
                self.log_gpu_usage(f"After loading VLLM model with {adapter_name}.")

        ######################################
        # Prepare evaluation model with vLLM
        ######################################
        elif self.eval_with == "hf":
            if self.hf_model is None:
                start_time = time.time()
                model_logger.info("Loading HF model for evaluation.")
                adapter_path = self.adapters[self.current_adapter_name]
                if adapter_path is None:
                    self.hf_model = AutoModelForCausalLM.from_pretrained(
                        **self.pretrained_args,
                        quantization_config=self.bits_and_bytes_configs
                    )
                    self.hf_model = get_peft_model(self.hf_model, self.lora_config)
                    model_logger.info(f"HF model prepared with new LoRA configuration.")
                else:
                    self.hf_model = AutoModelForCausalLM.from_pretrained(
                        **self.pretrained_args,
                        quantization_config=self.bits_and_bytes_configs
                    )
                    self.hf_model = PeftModel.from_pretrained(
                        model=self.hf_model,
                        model_id=adapter_path
                    )
                    model_logger.info(f"HF model loaded with LoRA weights from {adapter_path}.")
                self.hf_model.eval()
                end_time = time.time()
                compute_logger.info(f"HF model loading time: {end_time - start_time:.2f} seconds.")
                self.log_gpu_usage("After loading HF model.")

        ######################################
        # Prepare evaluation model with SGL
        ######################################
        elif self.eval_with == "sgl":
            if self.sgl_model is None:
                self.log_gpu_usage(f"Before loading SGL model with {adapter_name}.")
                start_time = time.time()
                import sgl
                self.sgl_model = sglang.LLM(
                    self.model_name,
                    use_adapter=True,
                    adapter_path=self.adapters[self.current_adapter_name],
                    seed=self.random_seed,
                    max_model_len=self.max_model_length,
                    device=self.device
                )
                self.sgl_model.to(self.device)
                end_time = time.time()
                compute_logger.info(f"SGL model loading time: {end_time - start_time:.2f} seconds.")
                self.log_gpu_usage(f"After loading SGL model with {adapter_name}.")

    def destroy_hf(self):
        """
        Destroys the Hugging Face model to free up memory.
        """
        model_logger.info("Destroying HF model.")
        if self.hf_model is not None:
            self.log_gpu_usage("Before destroying HF.")
            start_time = time.time()
            del self.hf_model
            gc.collect()
            torch.cuda.empty_cache()
            self.hf_model = None
            end_time = time.time()
            compute_logger.info(f"HF model unloading time: {end_time - start_time:.2f} seconds.")
            self.log_gpu_usage("After destroying HF.")


    def destroy_vllm(self):
        """
        Destroys the VLLM model to free up memory.
        """
        start_time = time.time()
        if self.vllm_model is not None:
            self.log_gpu_usage("Before destroying VLLM")
            del self.vllm_model
            gc.collect()
            torch.cuda.empty_cache()
            self.vllm_model = None
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            torch.cuda.device(0)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            self.log_gpu_usage("After destroying VLLM.")
        end_time = time.time()
        compute_logger.info(f"VLLM model unloading time: {end_time - start_time:.2f} seconds.")

    def log_gpu_usage(self, message: str) -> None:
        """
        Logs the GPU memory usage.
        Args:
            message (str): A message to include in the log.
        """
        gpu_memory = torch.cuda.memory_allocated() / (1024 ** 3)
        memory_logger.info(f"{message}: GPU memory allocated: {gpu_memory:.2f} GB")



    def prompt(self, contexts, kv_caches=None) -> any:
        """
        Generates a response (or responses) from the model based on the provided contexts.
        If kv_caches is provided (as a list with same length as contexts), generation will be performed iteratively
        for each context with its corresponding key-value cache, and the updated caches will be returned.
        """
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.device.index)
        adapter_path = self.adapters[self.current_adapter_name]
        if len(contexts) == 0:
            return []

        # If no kv_caches provided, use the existing batched generation approach
        if kv_caches is None:
            texts = self.tokenizer.apply_chat_template(
                contexts, tokenize=False, add_generation_prompt=True
            )
            start_time = time.time()

            ###############################
            # Generate response with vLLM
            ###############################
            if self.eval_with == "vllm":
                with torch.no_grad():
                    if adapter_path is not None:
                        model_logger.info(f"Generating using VLLM with LoRA at {adapter_path}")
                        self.vllm_id += 1
                        decoded = self.vllm_model.generate(
                            texts,
                            sampling_params=self.vllm_sampling_params,
                            lora_request=LoRARequest(f"dond_lora_{self.vllm_id}", self.vllm_id, adapter_path),
                        )
                    else:
                        model_logger.info("Generating using VLLM without LoRA")
                        decoded = self.vllm_model.generate(
                            texts, sampling_params=self.vllm_sampling_params
                        )
                responses = [d.outputs[0].text for d in decoded]
                del decoded

            ###############################
            # Generate responses with HF
            ###############################
            elif self.eval_with == "hf":
                if self.hf_model is None:
                    model_logger.error("HF model is not loaded. Cannot proceed with generation.")
                    return []
                with torch.no_grad():
                    model_logger.info("Generating using HF.")
                    self.log_gpu_usage("Before HF generation")
                    encoded_inputs = self.tokenizer(
                        texts, return_tensors="pt", padding=True, truncation=True
                    )
                    encoded_inputs = {k: v.to(self.device) for k, v in encoded_inputs.items()}
                    output = self.hf_model.generate(
                        **encoded_inputs,
                        max_new_tokens=self.hf_sampling_params["max_new_tokens"],
                        temperature=self.hf_sampling_params["temperature"],
                        top_k=self.hf_sampling_params["top_k"],
                        top_p=self.hf_sampling_params["top_p"],
                    )
                    responses = self.tokenizer.batch_decode(
                        output, skip_special_tokens=True
                    )
                    gc.collect()
                    torch.cuda.empty_cache()
                    self.log_gpu_usage("After HF generation")

            ###############################
            # Generate responses with SGL
            ###############################
            elif self.eval_with == "sgl":
                if self.sgl_model is None:
                    model_logger.error("SGL model is not loaded. Cannot proceed with generation.")
                    return []
                with torch.no_grad():
                    model_logger.info("Generating using SGL.")
                    self.log_gpu_usage("Before SGL generation")
                    encoded_inputs = self.tokenizer(
                        texts, return_tensors="pt", padding=True, truncation=True
                    )
                    encoded_inputs = {k: v.to(self.device) for k, v in encoded_inputs.items()}
                    output = self.sgl_model.generate(
                        **encoded_inputs,
                        max_new_tokens=self.sgl_sampling_params["max_new_tokens"],
                        temperature=self.sgl_sampling_params["temperature"],
                        top_k=self.sgl_sampling_params["top_k"],
                        top_p=self.sgl_sampling_params["top_p"]
                    )
                    responses = self.tokenizer.batch_decode(output, skip_special_tokens=True)
                    gc.collect()
                    torch.cuda.empty_cache()
                    self.log_gpu_usage("After SGL generation")
            else:
                model_logger.error(f"Unsupported generation method: {self.eval_with}")
                return []

            end_time = time.time()
            # Optionally log generation time
            return responses

        else:
            # Iterative generation with provided KV caches (one per context)
            if len(kv_caches) != len(contexts):
                model_logger.error("Length of kv_caches does not match number of contexts.")
                return []
            responses = []
            new_caches = []
            # Process each context individually, preserving its kv_cache
            for idx, context in enumerate(contexts):
                # Apply chat template for a single context
                text = self.tokenizer.apply_chat_template([context], tokenize=False, add_generation_prompt=True)[0]
                if self.eval_with == "hf":
                    with torch.no_grad():
                        model_logger.info("Generating using HF with KV cache.")
                        self.log_gpu_usage("Before HF generation (iterative)")
                        encoded_inputs = self.tokenizer(
                            text, return_tensors="pt", padding=False, truncation=True
                        )
                        encoded_inputs = {k: v.to(self.device) for k, v in encoded_inputs.items()}
                        # Pass the individual kv_cache
                        output = self.hf_model.generate(
                            **encoded_inputs,
                            max_new_tokens=self.hf_sampling_params["max_new_tokens"],
                            temperature=self.hf_sampling_params["temperature"],
                            top_k=self.hf_sampling_params["top_k"],
                            top_p=self.hf_sampling_params["top_p"],
                            past_key_values=kv_caches[idx],
                            use_cache=True,
                            return_dict_in_generate=True
                        )
                        # Determine how many tokens were input
                        input_length = encoded_inputs["input_ids"].shape[1]
                        sequence = output.sequences
                        completion = self.tokenizer.decode(sequence[0, input_length:], skip_special_tokens=True)
                        responses.append(completion)
                        # Retrieve the updated KV cache
                        new_caches.append(output.past_key_values)
                        gc.collect()
                        torch.cuda.empty_cache()
                        self.log_gpu_usage("After HF generation (iterative)")
                elif self.eval_with == "vllm":
                    # For vLLM, assume similar support; process iteratively
                    with torch.no_grad():
                        model_logger.info("Generating using VLLM with KV cache (iterative).")
                        # Expecting vLLM.generate to accept past_key_values, if not, fallback to batched
                        output = self.vllm_model.generate(
                            [text],
                            sampling_params=self.vllm_sampling_params,
                            lora_request=LoRARequest(f"dond_lora_{self.vllm_id}", self.vllm_id, adapter_path),
                            past_key_values=kv_caches[idx]
                        )
                        completion = output[0].outputs[0].text
                        responses.append(completion)
                        # For demonstration, assume updated cache is returned in the output (placeholder)
                        new_caches.append(kv_caches[idx])
                elif self.eval_with == "sgl":
                    with torch.no_grad():
                        model_logger.info("Generating using SGL with KV cache (iterative).")
                        encoded_inputs = self.tokenizer(
                            text, return_tensors="pt", padding=False, truncation=True
                        )
                        encoded_inputs = {k: v.to(self.device) for k, v in encoded_inputs.items()}
                        output = self.sgl_model.generate(
                            **encoded_inputs,
                            max_new_tokens=self.sgl_sampling_params["max_new_tokens"],
                            temperature=self.sgl_sampling_params["temperature"],
                            top_k=self.sgl_sampling_params["top_k"],
                            top_p=self.sgl_sampling_params["top_p"],
                            past_key_values=kv_caches[idx]  
                        )
                        input_length = encoded_inputs["input_ids"].shape[1]
                        completion = self.tokenizer.decode(output[0, input_length:], skip_special_tokens=True)
                        responses.append(completion)
                        new_caches.append(kv_caches[idx])
                else:
                    model_logger.error(f"Unsupported generation method: {self.eval_with}")
                    return []
            return responses, new_caches



    def export_current_adapter(self) -> None:
        """
        Saves only the LoRA weights to a specified directory. If the directory
        already exists, it deletes the existing directory before saving.
        """
        #self.hf_id += 1
        adapter_path = os.path.join(self.output_directory, f"{self.current_adapter_name}")

        # if os.path.exists(adapter_path):
        #     shutil.rmtree(adapter_path)
        #     logging.info(f"Existing directory '{adapter_path}' deleted.")

        os.makedirs(adapter_path, exist_ok=True)

        # Save only the LoRA weights
        if isinstance(self.hf_model, PeftModel) or isinstance(self.hf_model, AutoModelForCausalLMWithValueHead):
            self.hf_model.save_pretrained(adapter_path)
            model_logger.info(f"LoRA weights saved to {adapter_path}")
        else:
            model_logger.warning("Model is not a LoraModel or ValueHead, skipping LoRA weights saving.")

        # For vllm
        with open(os.path.join(adapter_path, "config.json"), "w") as f:
            json.dump({"model_type": "llama"}, f)

        # Update the adapter path after export
        self.adapters[self.current_adapter_name] = adapter_path
