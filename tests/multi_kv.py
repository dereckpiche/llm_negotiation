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

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(model_id)

user_prompts = ["Hello, what's your name?", "Tell me about bees."]

past_key_values = DynamicCache()
max_cache_length = past_key_values.get_max_cache_shape()

# Create initial convs
conversations = []
for prompt in user_prompts:
    conversations.append([{"role": "user", "content": prompt}])

# Add a list of varied follow-up prompts before the loop
follow_up_prompts = [
    "Please continue the conversation.",
    "What more can you share about the topic?",
    "Could you elaborate on that?",
]

# Talk in batch
for i in range(3):
    add_generation_prompt = True if i > 0 else False
    inputs = tokenizer.apply_chat_template(conversations, add_generation_prompt=add_generation_prompt, padding=True, truncation=True, return_tensors="pt", return_dict=True).to(model.device)
    output = model.generate(**inputs, do_sample=False, max_new_tokens=256, past_key_values=past_key_values)
    responses = tokenizer.batch_decode(output, skip_special_tokens=True)
    for (j, completion) in enumerate(responses):
        conversations[j].append({"role": "assistant", "content": completion})
    # Use a varied follow-up prompt based on the current iteration i
    follow_up_prompt = follow_up_prompts[i % len(follow_up_prompts)]
    for (j, _) in enumerate(responses):
        conversations[j].append({"role": "user", "content": follow_up_prompt})

for conv in conversations:
    print("-"*120)
    print("-"*120)
    print("-"*120)
    for msg in conv:
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')
        print(f"{role.capitalize()}: {content}")
    print("-"*120)
    print("-"*120)
    print("-"*120)


