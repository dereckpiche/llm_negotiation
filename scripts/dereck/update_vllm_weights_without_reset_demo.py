import os
os.environ["VLLM_USE_V1"] = '0'
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM
model_name = "google/gemma-3-1b-it"
if __name__ == "__main__":
    llm = LLM(model=model_name,
            enable_sleep_mode=True,
          worker_extension_cls='updatable_worker.UpdatableWorkerExtension')
    
    sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=100,
)

prompts = [
    "Hello, how are you?",
    "What is the capital of France?",
    "What is the capital of Germany?",
    "What is the capital of Italy?",
    "What is the capital of Spain?",
    "What is the capital of Portugal?",
    "What is the capital of Greece?",
    "What is the capital of Turkey?",
]

print("\n\n\n\*************GENERATING TEXT WITH THE ORIGINAL WEIGHTS! *************\n\n\n")
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    
model = AutoModelForCausalLM.from_pretrained(model_name)

llm.sleep()
llm.wake_up()

print("\n\n\n\*************MESSING WITH THE WEIGHTS! *************\n\n\n")
for name, param in model.named_parameters():
    llm.collective_rpc('update_weight', args=(name, param.data * 0.5)) # I just want to mess with the weights
    
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    
print("\n\n\n\*************FIXING THE WEIGHTS AGAIN! *************\n\n\n")
for name, param in model.named_parameters():
    llm.collective_rpc('update_weight', args=(name, param.data))
    
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")