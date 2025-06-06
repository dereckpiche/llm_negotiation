import os
from datetime import datetime

import pandas as pd
import torch
import torch.optim as optim
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from training.reinforce_trainer import ReinforceTrainerWRS
from training.reinforce_trainer_config import RtConfig

model_name = "Qwen/Qwen2.5-0.5B-Instruct" # "Qwen/Qwen2.5-0.5B-Instruct", "HuggingFaceTB/SmolLM-135M-Instruct", "arnir0/Tiny-LLM"

now = datetime.now()
logging_path = os.path.join(os.getcwd(), "tests/outputs_for_tests/{now}")
rt_config = RtConfig(
    entropy_coeff=0.01,
    kl_coeff=0.01,
    gradient_clipping=1.0,
    restrict_tokens=None,
    mini_batch_size=1,
    use_gradient_checkpointing=True,
    logging_path=logging_path,
    temperature=1.0,
    device="cuda:0",
    discount_factor=0.9,
    use_sum_rewards=False,
    ad_align_force_coop_first_step=True,
    use_advantage_alignment=True,
    use_variance_regularization_in_ad_align=True,
    use_time_regularization_in_ad_align=True,
    use_sign_in_ad_align=True,
    ad_align_clipping=5.0,
    ad_align_beta=1.0,
)



def test_get_training_data():
    all_contexts, all_scores, all_action_masks = trainer.get_training_data(
        "tests/inputs_for_tests/training_data_convs"
    )
    context = all_contexts[0]
    scores = all_scores[0]
    action_mask = all_action_masks[0]
    df = pd.DataFrame(
        data={
            "Token": tokenizer.convert_ids_to_tokens(context.tolist()),
            "Score": scores.tolist(),
            "Action Mask": action_mask.tolist(),
        }
    )
    df.to_csv("tests/outputs_for_tests/processed_conv.csv")


def test_train_on_folder():
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
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    trainer = ReinforceTrainerWRS(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=rt_config,
    )
    in_folder = "tests/inputs_for_tests/training_data_convs"
    files = os.listdir(in_folder)
    paths = [in_folder + "/" + file  for file in files]

    trainer.apply_reinforce_step_on_paths(
        paths
    )
    trainer.tally.save(path="tests/outputs_for_tests/tally_test_output/")

    print("Done")


if __name__ == "__main__":
    # test_new_reinforce_trainer()
    test_train_on_folder()
