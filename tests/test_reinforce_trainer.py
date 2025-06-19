import os
from datetime import datetime
from mllm.models.lean_local_llm import LeanLocalLLM

import pandas as pd
import torch
import torch.optim as optim
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from mllm.models.critic_wrapper import ScalarCritic

from mllm.training.reinforce_trainer import ReinforceTrainerWRS
from mllm.training.reinforce_trainer_config import RtConfig


now = datetime.now()
logging_path = os.path.join(os.getcwd(), "tests/outputs_for_tests/{now}")

in_folder = "tests/inputs_for_tests/training_data_convs"
files = os.listdir(in_folder)
paths = [in_folder + "/" + file  for file in files]

rt_config = RtConfig(
    entropy_coeff=0.0,
    kl_coeff=0.0,
    gradient_clipping=1.0,
    restrict_tokens=None,
    mini_batch_size=1,
    use_gradient_checkpointing=True,
    logging_path=logging_path,
    temperature=1.0,
    device="cuda:0",
    discount_factor=0.0,
    use_gae=True,
    gae_lambda=0.9,
    create_fake_bootstrap_value=False,
    use_sum_credits=False,
    use_advantage_alignment=False,
    ad_align_force_coop_first_step=True,
    ad_align_normalize_advantages=True,
    use_variance_regularization_in_ad_align=True,
    use_time_regularization_in_ad_align=True,
    use_sign_in_ad_align=True,
    ad_align_clipping=5.0,
    ad_align_beta=1.0,

    log_entropy_gradient_terms = True,
    log_kl_gradient_terms = True,
    log_value_gradient_terms = True,

    # Contextualized logging
    log_ctz_length = 30,
    log_ctz_top_k = 10,
    log_ctz_next_token = True,
    log_ctz_next_token_credit = True,
    log_ctz_next_token_log_prob = True,
    log_ctz_next_token_prob = True,
    log_ctz_top_k_tids = False,
    log_ctz_top_k_probs = False,
    log_ctz_top_slogpi = True,
    log_ctz_entropy = True,
    log_ctz_kl = True,
)

llm_config = {
    "max_model_length": 1e4,
    "device": "cuda",
    "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
    "generation_args": {
        "max_new_tokens": 120,
        "do_sample": True,
        "temperature": 0.7,
        "top_k": 0,
        "top_p": 1.0,
        "repetition_penalty": 1
    },
    "vllm_params": {
        "max_model_len": 13e3,
        "gpu_memory_utilization": 0.6,
        "enable_lora": True,
        "enable_prefix_caching": True,
        "enable_sleep_mode": True,
        "max_lora_rank": 64,
        "dtype": "bfloat16",
        "enforce_eager": True
    },
    "shared_hf_llm_init_kwargs": {
        "torch_dtype": "bfloat16",
        "device_map": "auto",
        "attn_implementation": "flash_attention_2"
    },
    "adapter_configs": {
        "self_play_agent": {
            "task_type": "CAUSAL_LM",
            "r": 64,
            "lora_alpha": 128,
            "lora_dropout": 0.0,
            "target_modules": "all-linear"
        },
        "self_play_critic": {
            "task_type": "CAUSAL_LM",
            "r": 32,
            "lora_alpha": 128,
            "lora_dropout": 0.0,
            "target_modules": "all-linear"
        }
    },
    "output_directory" : "tests/outputs_for_tests",
    "abort_vllm": True,
}


def test_train_on_folder():

    # Initialize Model
    shared_llm = LeanLocalLLM(**llm_config)
    adapters = shared_llm.get_adapter_pointers()
    policy = adapters["self_play_agent"]
    critic = ScalarCritic(adapters["self_play_critic"])
    policy_optimizer = optim.AdamW(policy.parameters(), lr=1e4)
    critic_optimizer = optim.AdamW(critic.parameters(), lr=1e4)

    # Get Trainer
    trainer = ReinforceTrainerWRS(
        model=policy,
        tokenizer=shared_llm.tokenizer,
        optimizer=policy_optimizer,
        lr_scheduler=None,
        critic=critic,
        critic_optimizer=critic_optimizer,
        critic_lr_scheduler=None,
        config=rt_config,
        save_path=""
    )

    trainer.set_training_data(paths=paths)
    shaping_info = trainer.send_trainer_info()
    trainer.use_co_trainer_info(shaping_info)
    trainer.train()
    trainer.tally.save(path="tests/outputs_for_tests/tally_test_output/")

    print("Done")


if __name__ == "__main__":
    # test_new_reinforce_trainer()
    test_train_on_folder()
