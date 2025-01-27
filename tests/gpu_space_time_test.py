import torch.nn.functional as F
from accelerate import Accelerator
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import os
import random
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import csv  # Add this import for CSV export
from peft import LoraConfig, get_peft_model  # Import necessary modules for LoRA

if __name__ == "__main__":

    # Initialize model and tokenizer with LoRA
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # Ensure dtype matches 'bfloat16'
        device_map="auto",  # Ensure device map matches 'auto'
        attn_implementation="flash_attention_2"  # Ensure attention implementation matches
    )

    # Configure LoRA
    lora_config = LoraConfig(
        r=64, 
        lora_alpha=32,  
        lora_dropout=0,  
        target_modules="all-linear"  
    )

    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)
    model.train()
    model.gradient_checkpointing_enable(dict(use_reentrant=False))
   

    # Initialize Accelerator for mixed precision
    accelerator = Accelerator()

    # Create optimizer for LoRA parameters only
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # Prepare model and optimizer with accelerator
    model, optimizer = accelerator.prepare(model, optimizer)
    torch.cuda.empty_cache()

    # Generate contexts of different lengths
    lengths = range(1, 50001, 200)
    memory_usages = []
    time_usages = []  # To store time taken for each iteration

    for length in lengths:
        start_time = time.time()  # Start timing

        # Create a dummy context of the specified length
        context = torch.randint(0, tokenizer.vocab_size, (length,)).unsqueeze(0).to('cuda')
        returns = torch.ones(length).unsqueeze(0).to('cuda')
        mask = torch.ones(length).unsqueeze(0).to('cuda')

        # Forward pass without checkpointing
        outputs = model(context)  # Directly call the model
        logits = outputs.logits

        # Compute dummy loss
        log_probs = F.log_softmax(logits, dim=-1)
        action_log_probs = log_probs.gather(dim=-1, index=context.unsqueeze(-1)).squeeze(-1)
        rewarded_action_log_probs = action_log_probs * (returns * mask)
        loss = -rewarded_action_log_probs.mean()

        # Backward pass using accelerator
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.empty_cache()

        # Clear memory cache after backward pass

        # Measure memory usage
        memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert to GB
        max_memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)  # Convert to GB
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # Convert to GB
        memory_percentage = (memory_allocated / total_memory) * 100

        memory_usages.append((length, memory_allocated, max_memory_allocated, memory_percentage))

        # Measure time taken
        end_time = time.time()
        time_taken = end_time - start_time
        time_usages.append((length, time_taken))

        # Plotting max memory usage after each iteration
        lengths, max_memory_allocated = zip(*[(l, m) for l, _, m, _ in memory_usages])
        plt.figure(figsize=(20, 8))  # Make the plot wider
        plt.plot(lengths, max_memory_allocated, label='Max Memory Allocated (GB)', linestyle='--', marker='x')
        
        # Annotate each point with its value
        for i, (x, y) in enumerate(zip(lengths, max_memory_allocated)):
            plt.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

        plt.xlabel('Context Length')
        plt.ylabel('Max Memory Usage (GB)')
        plt.title('Max GPU Memory Usage vs. Context Length')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(os.path.dirname(__file__), 'gpu_max_memory_usage.png'))
        plt.close()

        # Plotting time usage after each iteration
        lengths, time_taken = zip(*time_usages)
        plt.figure(figsize=(20, 8))  # Make the plot wider
        plt.plot(lengths, time_taken, label='Time Taken (s)', marker='o')
        
        # Annotate each point with its value
        for i, (x, y) in enumerate(zip(lengths, time_taken)):
            plt.annotate(f'{y:.0f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

        plt.xlabel('Context Length')
        plt.ylabel('Time Taken (s)')
        plt.title('Time Taken vs. Context Length')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(os.path.dirname(__file__), 'time_taken.png'))
        plt.close()

        # Reset max memory allocated
        torch.cuda.reset_max_memory_allocated()

    # Clean up
    del model
    torch.cuda.empty_cache()