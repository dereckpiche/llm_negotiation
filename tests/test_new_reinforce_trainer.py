import os

import pandas as pd
import torch
import torch.optim as optim
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from training.new_reinforce_trainer import ReinforceTrainerWRS
from training.reinforce_trainer_config import RtConfig


def test_simple_step():
    # Use a tiny model for testing
    model_name = "HuggingFaceTB/SmolLM-135M-Instruct"  # or arnir0/Tiny-LLM
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

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    config = RtConfig(
        entropy_coeff=0.01,
        kl_coeff=0.01,
        gradient_clipping=1.0,
        top_k_for_logging=3,
        restrict_tokens=None,
        mini_batch_size=2,
        use_gradient_checkpointing=False,
        logging_path=os.path.join(os.getcwd(), "test_log"),
        temperature=1.0,
        device="cuda:0",
    )

    trainer = ReinforceTrainer(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=config,
    )

    # Dummy data
    contexts = [
        torch.tensor([60, 87, 69, 98, 96, 69]),
        torch.tensor([35, 45, 65, 76, 64, 10]),
    ]
    scores = [
        torch.tensor([0.0, 0.3, 0.0, 1.0, 1.0, 0.0]),
        torch.tensor([0.0, 30.3, 0.0, 20.2, 0.0, 10.1]),
    ]
    action_masks = [torch.tensor([0, 1, 0, 1, 1, 0]), torch.tensor([0, 1, 0, 1, 0, 1])]

    # Run a reinforce step
    trainer.apply_reinforce_step(contexts, scores, action_masks)
    tally = trainer.tally
    import pdb

    print("Reinforce step completed.")


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
    model_name = "HuggingFaceTB/SmolLM-135M-Instruct"  # or arnir0/Tiny-LLM
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

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    from datetime import datetime

    now = datetime.now()
    logging_path = os.path.join(os.getcwd(), "tests/outputs_for_tests/{now}")

    config = RtConfig(
        entropy_coeff=0.01,
        kl_coeff=0.01,
        gradient_clipping=1.0,
        top_k_for_logging=3,
        restrict_tokens=None,
        mini_batch_size=2,
        use_gradient_checkpointing=False,
        logging_path=logging_path,
        temperature=1.0,
        device="cuda:0",
        discount_factor=0.9,
        use_sum_rewards=False,
        use_advantage_alignment=False,
    )

    trainer = ReinforceTrainerWRS(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=config,
    )

    trainer.apply_reinforce_step_on_data_folder(
        "tests/inputs_for_tests/training_data_convs"
    )

    print("Done")


if __name__ == "__main__":
    # test_new_reinforce_trainer()
    test_train_on_folder()
