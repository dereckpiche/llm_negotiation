import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def build_prompt(context):
    """
    Build a simple prompt string from a conversation context.
    Each element in context is a dict with keys 'role' and 'content'.
    """
    prompt = ""
    for msg in context:
        if msg["role"] == "user":
            prompt += f"User: {msg['content']}\n"
        else:
            prompt += f"Assistant: {msg['content']}\n"
    prompt += "Assistant: "  # Signal for the assistant to generate a response.
    return prompt

def initialize_cache(model, input_ids, attention_mask):
    """
    Run a forward pass to obtain initial past_key_values.
    """
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
    return outputs.past_key_values

def main():
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # 8B model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load tokenizer and model using bfloat16.
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Ensure padding token is set.
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.to(device)

    # Define several conversation contexts.
    contexts = [
        [{"role": "user", "content": "Hello, what's your name?"}],
        [{"role": "user", "content": "Tell me your thoughts about space travel."}],
        [{"role": "user", "content": "What's the capital of France?"}],
        [{"role": "user", "content": "Share a fun fact about animals."}],
    ]
    
    # Build prompts for each conversation.
    prompts = [build_prompt(ctx) for ctx in contexts]
    
    # Tokenize all prompts as a batch.
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Initialize a separate KV cache for each prompt.
    # For each prompt, tokenize without padding/truncation and perform a forward pass.
    kv_caches = []
    for prompt in prompts:
        single_inputs = tokenizer(prompt, return_tensors="pt").to(device)
        past = initialize_cache(model, single_inputs.input_ids, single_inputs.attention_mask)
        kv_caches.append(past)
    
    # Combine individual caches into a batched past_key_values structure.
    # Each element in past_key_values is a tuple (key, value) per transformer layer.
    # We stack the keys and values along the batch dimension.
    batched_past = []
    num_layers = len(kv_caches[0])
    batch_size = len(kv_caches)
    for layer in range(num_layers):
        # Each kv_caches[i][layer] is a tuple of (key, value) for a single sample.
        keys = torch.cat([kv_caches[i][layer][0] for i in range(batch_size)], dim=0)
        values = torch.cat([kv_caches[i][layer][1] for i in range(batch_size)], dim=0)
        batched_past.append((keys, values))
    
    print("Generating responses in batch using combined KV caches...")
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        temperature=0.7,
        use_cache=True,
        past_key_values=batched_past,  # Batched past key values: one per prompt.
        return_dict_in_generate=True
    )
    
    # Decode and print responses.
    responses = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
    for i, response in enumerate(responses):
        print(f"Response {i}: {response}")

if __name__ == "__main__":
    main() 