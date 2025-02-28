import torch
from transformers import AutoTokenizer,AutoModelForCausalLM,GenerationConfig
from transformers.cache_utils import (
   DynamicCache,
   SinkCache,
   StaticCache,
   SlidingWindowCache,
   QuantoQuantizedCache,
   QuantizedCacheConfig,
)
import time
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map='cuda:0')
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

def test(use_cache=True):

    if use_cache:
        past_key_values = DynamicCache()

    time_start = time.time()
    user_prompts = ["Tell me about the central limit theorem."]


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

        ttg = 400
        gen_config = GenerationConfig(min_new_tokens=ttg, max_new_tokens=ttg)
        if use_cache:
            outputs = model.generate(**inputs, do_sample=True, generation_config=gen_config, past_key_values=past_key_values)
        else:
            outputs = model.generate(**inputs, do_sample=True, generation_config=gen_config)

        responses = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        for (j, completion) in enumerate(responses):
            conversations[j].append({"role": "assistant", "content": completion})
        # Use a varied follow-up prompt based on the current iteration i
        follow_up_prompt = follow_up_prompts[i % len(follow_up_prompts)]
        for (j, _) in enumerate(responses):
            conversations[j].append({"role": "user", "content": follow_up_prompt})


    # for conv in conversations:
    #     print("-"*120)
    #     print("-"*120)
    #     print("-"*120)
    #     for msg in conv:
    #         role = msg.get('role', 'unknown')
    #         content = msg.get('content', '')
    #         print(f"{role.capitalize()}: {content}")
    #     print("-"*120)
    #     print("-"*120)
    #     print("-"*120)
    time_end = time.time()
    return time_end - time_start



if __name__ == "__main__":
    time_no_cache = test(use_cache=False)
    time_cache = test(use_cache=True)
    print(f"Time without cache: {time_no_cache} seconds")
    print(f"Time with cache: {time_cache} seconds")
