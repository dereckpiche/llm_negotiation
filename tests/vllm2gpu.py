import torch
from vllm import LLM
import os 
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
from vllm.sampling_params import SamplingParams
from transformers import GenerationConfig

device0 = torch.device("cuda:0")
device1 = torch.device("cuda:1")

# Initialize the first agent using vLLM
llm_agent1 = LLM("meta-llama/Meta-Llama-3.1-8B-Instruct", dtype=torch.bfloat16, enable_lora=False, max_model_len=5000, tensor_parallel_size=1, device=device0)

# Initialize the second agent using Hugging Face Transformers
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
# Ensure a pad_token is set to avoid error on padding
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

llm_agent2 = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device1)
llm_agent2.eval()

prompt = "Once upon a time"
num_prompts = 5
prompts = [prompt] * num_prompts

ttg = 2000  # tokens to generate

# Measure generation time for the first agent (vLLM)
start_time = time.time()
output1 = llm_agent1.generate(prompts, sampling_params=SamplingParams(max_tokens=ttg, ignore_eos=True))
end_time = time.time()
vllm_time = end_time - start_time

# Measure generation time for the second agent using HF Transformers
start_time = time.time()
input_ids = tokenizer(prompts, return_tensors="pt", padding=True).input_ids.to(device1)
output2 = llm_agent2.generate(input_ids, generation_config=GenerationConfig(min_new_tokens=ttg, max_new_tokens=ttg))
end_time = time.time()
hf_time = end_time - start_time

# Process and print token lengths for each generated prompt for vLLM
print("vLLM Agent Token Lengths:")
for i, out in enumerate(output1):
    agent1_tokens = tokenizer(out.outputs[0].text, return_tensors="pt").input_ids
    print(f"Prompt {i+1} len (tokens):", agent1_tokens.size(1))

# Process and print token lengths for each generated prompt for HF Transformers
print("HF Transformers Agent Token Lengths:")
# output2 is a batch of sequences
for i, seq in enumerate(output2):
    output_text = tokenizer.decode(seq)
    output_tokens = tokenizer(output_text, return_tensors="pt").input_ids
    print(f"Prompt {i+1} len (tokens):", output_tokens.size(1))

print("Agent 1 Generation Time:", vllm_time, "seconds")
print("Agent 2 Generation Time:", hf_time, "seconds")