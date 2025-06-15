import os
from datetime import datetime

import pandas as pd
import torch
import torch.optim as optim
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from models.critic_wrapper import ScalarCritic

from training.reinforce_trainer import ReinforceTrainerWRS
from training.reinforce_trainer_config import RtConfig

model_name = "Qwen/Qwen2.5-0.5B-Instruct" # "Qwen/Qwen2.5-0.5B-Instruct", "HuggingFaceTB/SmolLM-135M-Instruct", "arnir0/Tiny-LLM"

now = datetime.now()
logging_path = os.path.join(os.getcwd(), "tests/outputs_for_tests/{now}")

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
    use_gae=False,
    gae_lambda=0.0,
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


def test_train_on_folder():

    # Get Input Json Paths
    in_folder = "tests/inputs_for_tests/training_data_convs"
    files = os.listdir(in_folder)
    paths = [in_folder + "/" + file  for file in files]

    # Initialize Model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules="all-linear",
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    optimizer = optim.AdamW(model.parameters(), lr=1e4)
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=10, 
        gamma=0.9)

    critic = ScalarCritic(get_peft_model(model, lora_config)).to("cuda:0")
    critic_optimizer = optim.AdamW(critic.parameters(), lr=1e4)
    critic_lr_scheduler = optim.lr_scheduler.StepLR(
        critic_optimizer, 
        step_size=10, 
        gamma=0.9)
    


    # Get Trainer
    trainer = ReinforceTrainerWRS(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        critic=critic,
        critic_optimizer=critic_optimizer,
        critic_lr_scheduler=critic_lr_scheduler,
        config=rt_config,
    )

    trainer.set_training_data(paths=paths)
    shaping_info = trainer.send_shaping_info_to_opponents()
    trainer.use_opponents_shaping_info(shaping_info)
    trainer.train()
    trainer.tally.save(path="tests/outputs_for_tests/tally_test_output/")

    print("Done")


if __name__ == "__main__":
    # test_new_reinforce_trainer()
    test_train_on_folder()
