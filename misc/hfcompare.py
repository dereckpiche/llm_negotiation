import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from transformers.cache_utils import (
   DynamicCache,
   SinkCache,
   StaticCache,
   SlidingWindowCache,
   QuantoQuantizedCache,
   QuantizedCacheConfig,
)
import time

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map='cuda')
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_id)

user_prompts = [
    "Hello, what's your name?",
    "Can you tell me about yourself?",
    "What do you do for fun?",
    "Do you enjoy music?",
    "What's your favorite song?",
    "Tell me a joke.",
    "What do you think about AI?",
    "How do you compute temperature?",
    "What is quantum computing?",
    "Let's discuss the future of technology."
]  # Simulated longer conversation

past_key_values = DynamicCache()
max_cache_length = past_key_values.get_max_cache_shape()

# function to run generation experiment with optional cache

def run_experiment(use_cache):
    # reset messages for each experiment
    messages = []
    # if using cache, create one, otherwise leave as None
    cache = DynamicCache() if use_cache else None
    
    for prompt in user_prompts:
        messages.append({"role": "user", "content": prompt})
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
        input_length = inputs["input_ids"].shape[1]
        if cache is not None:
            outputs = model.generate(**inputs, do_sample=False, max_new_tokens=256, past_key_values=cache)
        else:
            outputs = model.generate(**inputs, do_sample=False, max_new_tokens=256)
        completion = tokenizer.decode(outputs[0, input_length: ], skip_special_tokens=True)
        messages.append({"role": "assistant", "content": completion})
    return messages

# Timing experiment with chat cache
start_time_cache = time.time()
messages_with_cache = run_experiment(True)
time_with_cache = time.time() - start_time_cache

# Timing experiment without chat cache
start_time_no_cache = time.time()
messages_without_cache = run_experiment(False)
time_without_cache = time.time() - start_time_no_cache

print("Time with chat cache:", time_with_cache)
print("Time without chat cache:", time_without_cache)
