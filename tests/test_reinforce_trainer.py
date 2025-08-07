import os
from datetime import datetime

import torch.optim as optim

from mllm.markov_games.gather_and_export_utils import *
from mllm.models.large_language_model_local import LeanLocalLLM
from mllm.models.scalar_critic import ScalarCritic
from mllm.training.reinforce_trainer import BaseTrainer

now = datetime.now()
logging_path = os.path.join(os.getcwd(), "tests/outputs_for_tests/{now}")


trainer_kwargs = {
    "enable_tokenwise_logging": True,
    "entropy_coeff": 0.0,
    "kl_coeff": 0.0,
    "gradient_clipping": 1.0,
    "restrict_tokens": None,
    "mini_batch_size": 1,
    "use_gradient_checkpointing": False,
    "temperature": 0.7,
    "device": "cuda",
    "use_gae": False,
    "use_rloo": False,
    "gae_lambda_for_credits": 0.0,
    "gae_lambda_for_targets": 0.0,
    "discount_factor": 0.99,
    "reward_normalizing_constant": 1.0,
}

llm_config = {
    "llm_id": "base_llm",
    "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
    "inference_backend": "dummy",
    "hf_kwargs": {
        "device_map": "auto",
        "torch_dtype": "bfloat16",
        "max_memory": {0: "15GiB"},
        "attn_implementation": "flash_attention_2",
    },
    "inference_backend_init_kwargs": {},
    "inference_backend_sampling_params": {},
    "adapter_configs": {
        "agent_adapter": {
            "task_type": "CAUSAL_LM",
            "r": 32,
            "lora_alpha": 128,
            "lora_dropout": 0.0,
            "target_modules": "all-linear",
        },
        "critic_adapter": {
            "task_type": "CAUSAL_LM",
            "r": 32,
            "lora_alpha": 128,
            "lora_dropout": 0.0,
            "target_modules": "all-linear",
        },
    },
}

# Initialize Model
shared_llm = LeanLocalLLM(**llm_config)
adapters = shared_llm.get_adapter_modules()
policy = adapters["agent_adapter"]
critic = ScalarCritic(adapters["critic_adapter"])
policy_optimizer = optim.AdamW(policy.parameters(), lr=1e4)
critic_optimizer = optim.AdamW(critic.parameters(), lr=1e4)

# Get Trainer
trainer = BaseTrainer(
    policy=policy,
    tokenizer=shared_llm.tokenizer,
    policy_optimizer=policy_optimizer,
    lr_scheduler=None,
    critic=critic,
    critic_optimizer=critic_optimizer,
    critic_lr_scheduler=None,
    save_path="tests/outputs_for_tests/base_trainer_test_output/",
    **trainer_kwargs,
)
input_folder = "tests/inputs_for_tests/"


rollout_trees = gather_rollout_trees(input_folder)
trainer.set_policy_gradient_data(
    rollout_trees=rollout_trees, agent_ids=["Alice", "Bob"]
)
trainer.train()
trainer.tally.save(path="tests/outputs_for_tests/tally_test_output/")
trainer.tokenwise_tally.save(
    path="tests/outputs_for_tests/tokenwise_tally_test_output/"
)

print("Done")
