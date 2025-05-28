import os

import torch
import torch.optim as optim
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from training.new_reinforce_trainer import ReinforceTrainer
from training.reinforce_trainer_config import RtConfig


def test_new_reinforce_trainer():
    # Use a tiny model for testing
    model_name = "arnir0/Tiny-LLM"
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
        torch.tensor([50256, 1, 2, 3, 4, 5]),
        torch.tensor([50256, 6, 7, 8, 9, 10]),
    ]
    scores = [
        torch.tensor([0.0, 0.3, 0.0, 1.0, 1.0, 0.0]),
        torch.tensor([0.0, 30.3, 0.0, 20.2, 0.0, 10.1]),
    ]
    action_masks = [torch.tensor([0, 1, 0, 1, 1, 0]), torch.tensor([0, 1, 0, 1, 0, 1])]

    # Run a reinforce step
    trainer.apply_reinforce_step(contexts, scores, action_masks)
    print("Reinforce step completed.")


if __name__ == "__main__":
    test_new_reinforce_trainer()
