import torch
import os
import time
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

if __name__ == "__main__":

    # Initialize model and tokenizer
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token

    vllm_sampling_params = SamplingParams(
        temperature=1.0,  # Example temperature, adjust as needed
        top_k=50,         # Example top_k, adjust as needed
        top_p=0.95,       # Example top_p, adjust as needed
        max_tokens=1,      # Generate one more token
        ignore_eos=True
    )

    # Load VLLM model
    model = LLM(model_name, enable_lora=False, max_model_len=50000)

    # Generate contexts of different lengths
    lengths = range(1, 50001, 200)
    time_usages = []  # To store time taken for each generation

    for length in lengths:
        # Create a dummy text context of the specified length
        vllm_sampling_params.max_tokens = length + 1

        # Measure generation time
        start_time = time.time()
        outputs = model.generate("Yo", sampling_params=vllm_sampling_params)
        end_time = time.time()

        time_taken = end_time - start_time
        time_usages.append((length, time_taken))

        # Plotting time usage after each iteration
        lengths, time_taken = zip(*time_usages)
        plt.figure(figsize=(20, 8))  # Make the plot wider
        plt.plot(lengths, time_taken, label='Time Taken (s)', marker='o')
        
        # Annotate each point with its value
        for i, (x, y) in enumerate(zip(lengths, time_taken)):
            plt.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

        plt.xlabel('Context Length')
        plt.ylabel('Time Taken (s)')
        plt.title('Generation Time vs. Context Length')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(os.path.dirname(__file__), 'generation_time.png'))
        plt.close()

    # Clean up
    del model
    torch.cuda.empty_cache()