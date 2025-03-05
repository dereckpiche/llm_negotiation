import torch
from vllm import LLM
 
# Explicitly create torch device objects
 
# Create two separate LLM instances.
# (If your model fits on a single GPU, set tensor_parallel_size=1.
# For larger models, adjust tensor_parallel_size accordingly.)
device0 = torch.device("cuda:0")
llm_agent1 = LLM("meta-llama/Meta-Llama-3.1-8B-Instruct", dtype=torch.bfloat16, enable_lora=True, max_model_len=1000, tensor_parallel_size=1, device=device0)

device1 = torch.device("cuda:1")
llm_agent2 = LLM("meta-llama/Meta-Llama-3.1-8B-Instruct", dtype=torch.bfloat16, enable_lora=True, max_model_len=1000, tensor_parallel_size=1, device=device1)
 
# Now you can use each agent independently:
prompt = "Once upon a time"
output1 = llm_agent1.generate([prompt])
output2 = llm_agent2.generate([prompt])
 
print("Agent 1:", output1[0].outputs[0].text)
print("Agent 2:", output2[0].outputs[0].text)