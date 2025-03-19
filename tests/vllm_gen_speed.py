import torch
from vllm import *
 
device0 = torch.device("cuda:0")
llm_agent1 = LLM("meta-llama/Meta-Llama-3.1-8B-Instruct", dtype=torch.bfloat16, enable_lora=True, max_model_len=13000, gpu_memory_utilization=0.3, device=device0)

llm_agent2 = LLM("meta-llama/Meta-Llama-3.1-8B-Instruct", dtype=torch.bfloat16, enable_lora=True, max_model_len=13000, gpu_memory_utilization=0.3, device=device0)

# Now you can use each agent independently:
prompt = "Once upon a time"
import time 
t_start = time.time()
output1 = llm_agent1.generate([prompt*32], sampling_params=SamplingParams(skip_special_tokens=True, max_tokens=13000))
output2 = llm_agent2.generate([prompt*32], sampling_params=SamplingParams(skip_special_tokens=True, max_tokens=13000))
t_end = time.time()
print(f"Time taken: {(t_end - t_start)/60} minutes")
