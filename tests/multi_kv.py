import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache

# Use your Llama 3.1 8B model id here.
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # replace with the correct model repo id if needed
device = "cuda:0"

# load model and tokenizer using bfloat16 on cuda:0
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map={'': device}  # forces the model on cuda:0
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Suppose we have two conversations, each with initial messages
conversations = [
    [{"role": "user", "content": "Hello, what's your name?"}],
    [{"role": "user", "content": "Tell me a fun fact."}],
]

# In our sequential example we used a single past_key_values for the conversation.
# For batch processing, we create one cache per conversation.
# (Note: for proper batching, these caches need to support being "stacked" or merged into one batched structure.
#  You may need to adapt DynamicCache accordingly.)
individual_caches = [DynamicCache() for _ in conversations]

# New user turns for these conversations (processing the batch simultaneously)
user_prompts = ["Hi, how are you?", "And what's your favorite color?"]

# Prepare inputs for each conversation turn
inputs_list = []
input_lengths = []
for conv, prompt in zip(conversations, user_prompts):
    # Append the new user message
    conv.append({"role": "user", "content": prompt})
    # Create the input from the full chat history using the chat template helper.
    # This function is expected to embed the conversation history into a prompt.
    inputs = tokenizer.apply_chat_template(
        conv,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    )
    inputs_list.append(inputs)
    input_lengths.append(inputs["input_ids"].shape[1])

# Batch the inputs (pad sequence if needed)
# Use a fallback value (0) if tokenizer.pad_token_id is None
pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

batch_input_ids = torch.nn.utils.rnn.pad_sequence(
    [inp["input_ids"].squeeze(0) for inp in inputs_list],
    batch_first=True,
    padding_value=pad_id
).to(device)
batch_attention_mask = torch.nn.utils.rnn.pad_sequence(
    [inp["attention_mask"].squeeze(0) for inp in inputs_list],
    batch_first=True,
    padding_value=0
).to(device)

batched_inputs = {"input_ids": batch_input_ids, "attention_mask": batch_attention_mask}

# Here we try to combine each conversation's KV cache into one batched KV cache.
# For illustration we assume that DynamicCache exposes an attribute (e.g. ".tensor") that holds its internal cached tensors.
# You might need to implement your own merging in your DynamicCache class.
batched_past_key_values = None
if all(hasattr(cache, "tensor") for cache in individual_caches):
    # Stack the saved tensors along a new dimension corresponding to the batch.
    cached_tensors = [cache.tensor for cache in individual_caches]
    batched_past_key_values = torch.stack(cached_tensors, dim=0)
    # NOTE: The resulting structure must be in the format that model.generate expects for past_key_values.
    # This may require custom modifications beyond simple stacking.

# Now generate outputs in batch. (If batched_past_key_values is None then generation will start from scratch.)
outputs = model.generate(
    **batched_inputs,
    do_sample=False,
    max_new_tokens=256,
    past_key_values=batched_past_key_values
)

# Decode results for each conversation and append the assistant messages
for i, input_length in enumerate(input_lengths):
    generated_tokens = outputs[i, input_length:]
    completion = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    conversations[i].append({"role": "assistant", "content": completion})

# Print the conversations with separators and clear turn indicators
separator = "_" * 80
for idx, conv in enumerate(conversations):
    print(separator)
    print(f"Conversation {idx+1}")
    for turn, message in enumerate(conv):
        role = message["role"].capitalize()
        print(f"{role} (Turn {turn+1}): {message['content']}\n")
print(separator)