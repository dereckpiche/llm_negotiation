import torch
from vllm import *
import time
import multiprocessing
import os
import numpy as np
from transformers import AutoModelForCausalLM

# see https://docs.vllm.ai/en/latest/serving/engine_args.html
vllm_params = {
    "model": "google/gemma-3-4b-it",
    "dtype": torch.bfloat16,
    "enable_lora": True,
    "max_model_len": 13000,
    "gpu_memory_utilization": 0.9,
    "device": torch.device("cuda:0"),
    "enforce_eager": False,  # Default False
    "block_size": 128,  # Default: 128
    "enable_prefix_caching": True,
    "disable_custom_all_reduce": False,  # Default False
    "seed": 42,
    "enable_chunked_prefill": False,  # Default False
    "worker_extension_cls": "updatable_worker.UpdatableWorkerExtension"  # Add extension for weight updates
}

# Set up experiment parameters
total_games = 32
parallel_games = 32
nb_rounds = 4
nb_turns = 2
intro_prompt_length = 800
intermediary_length = 200
response_lengths = [100, 20]
num_inference_iterations = 3  # Define number of iterations

# Function to run vLLM on a specific GPU
def run_vllm_process(gpu_id, queue_in, queue_out):
    # Set environment variable to restrict GPU visibility
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Create vLLM instance with modified parameters for this GPU
    local_vllm_params = vllm_params.copy()
    local_vllm_params["device"] = torch.device("cuda:0")  # Always cuda:0 in isolated process
    
    # Initialize vLLM
    llm_agent = LLM(**local_vllm_params)
    
    # Wait for and process requests
    while True:
        task = queue_in.get()
        if task is None:  # Poison pill to terminate
            break
            
        task_type = task[0]
        
        if task_type == "generate":
            _, task_id, contexts, max_tokens = task
            
            # Generate responses
            responses = llm_agent.generate(contexts, sampling_params=SamplingParams(
                skip_special_tokens=True, 
                max_tokens=max_tokens
            ))
            responses = [r.outputs[0].text for r in responses]
            
            # Send results back
            queue_out.put((task_id, responses))
        
        elif task_type == "sleep":
            llm_agent.sleep()
            queue_out.put(("sleep_done", None))
            
        elif task_type == "wake":
            llm_agent.wake_up()
            queue_out.put(("wake_done", None))
            
        elif task_type == "update_weight":
            _, name, param = task
            llm_agent.collective_rpc('update_weight', args=(name, param))
            queue_out.put(("weight_updated", name))



# Create initial contexts
full_contexts_1 = [str(i) + intro_prompt_length*"0" for i in range(total_games)]
full_contexts_2 = [str(i) + intro_prompt_length*"1" for i in range(total_games)]

# Create queues for inter-process communication
q1_in = multiprocessing.Queue()
q1_out = multiprocessing.Queue()
q2_in = multiprocessing.Queue()
q2_out = multiprocessing.Queue()

# Start processes on separate GPUs
p1 = multiprocessing.Process(target=run_vllm_process, args=(0, q1_in, q1_out))
p2 = multiprocessing.Process(target=run_vllm_process, args=(1, q2_in, q2_out))

p1.start()
p2.start()

stime = time.time()

# Initialize two HF model instances on CPU
print("Initializing HuggingFace models on CPU")
hf_model1 = AutoModelForCausalLM.from_pretrained(vllm_params["model"], torch_dtype=vllm_params["dtype"])
hf_model1.to("cpu")
hf_model2 = AutoModelForCausalLM.from_pretrained(vllm_params["model"], torch_dtype=vllm_params["dtype"])
hf_model2.to("cpu")

for k in range(num_inference_iterations):
    print(f"Starting inference iteration {k+1}/{num_inference_iterations}")

    for i in range(0, int(total_games/parallel_games)):
        contexts_1 = full_contexts_1[i*parallel_games: (i+1)*parallel_games]
        contexts_2 = full_contexts_2[i*parallel_games: (i+1)*parallel_games]
        
        for j in range(nb_rounds):
            for k in range(nb_turns):
                # Send requests to both processes
                q1_in.put(("generate", f"r{j}t{k}", contexts_1, response_lengths[k]))
                _, responses_1 = q1_out.get()

                q2_in.put(("generate", f"r{j}t{k}", contexts_2, response_lengths[k]))
                _, responses_2 = q2_out.get()   
                
                # Update contexts
                contexts_1 = [c+r for c,r in zip(contexts_1, responses_1)]
                contexts_1 = [c+intermediary_length*"0" for c in contexts_1]
                
                contexts_2 = [c+r for c,r in zip(contexts_2, responses_2)]
                contexts_2 = [c+intermediary_length*"1" for c in contexts_2]
    
    # Put vLLM instances to sleep and log memory usage
    print("Putting vLLM instances to sleep")
    before_sleep_memory = torch.cuda.memory_allocated(0) / (1024**3)  # Memory in GB
    q1_in.put(("sleep", None))
    q1_out.get()  # Wait for sleep to complete
    
    after_sleep_memory_gpu0 = torch.cuda.memory_allocated(0) / (1024**3)
    print(f"GPU 0 memory before sleep: {before_sleep_memory:.2f} GB, after sleep: {after_sleep_memory_gpu0:.2f} GB")
    
    before_sleep_memory = torch.cuda.memory_allocated(1) / (1024**3)
    q2_in.put(("sleep", None))
    q2_out.get()  # Wait for sleep to complete
    
    after_sleep_memory_gpu1 = torch.cuda.memory_allocated(1) / (1024**3)
    print(f"GPU 1 memory before sleep: {before_sleep_memory:.2f} GB, after sleep: {after_sleep_memory_gpu1:.2f} GB")

    # Move HuggingFace instances to respective GPUs
    print("Moving HuggingFace models to GPUs")
    hf_model1.to("cuda:0")
    hf_model2.to("cuda:1")
    
    # Simulate some work with HF models
    print("Running some operations with HuggingFace models on GPUs")
    time.sleep(5)  # Simulate work

    # Move HuggingFace models back to CPU
    print("Moving HuggingFace models back to CPU")
    hf_model1.to("cpu")
    hf_model2.to("cpu")

    # Transfer HF model weights to vLLM instances
    print("Transferring weights from HuggingFace models to vLLM instances")
    for name, param in hf_model1.named_parameters():
        q1_in.put(("update_weight", name, param.data))
        q1_out.get()  # Wait for weight update to complete
        
    for name, param in hf_model2.named_parameters():
        q2_in.put(("update_weight", name, param.data))
        q2_out.get()  # Wait for weight update to complete

    # Wake up the vLLM instances
    print("Waking up vLLM instances")
    q1_in.put(("wake", None))
    q1_out.get()  # Wait for wake to complete
    
    q2_in.put(("wake", None))
    q2_out.get()  # Wait for wake to complete

# Send poison pills to terminate processes
q1_in.put(None)
q2_in.put(None)

# Join processes
p1.join()
p2.join()

etime = time.time()
print(f"Experiment 4 took {(etime-stime)/60} minutes.")

tokens_generated = total_games * nb_rounds * sum(response_lengths)
print(f"Each model ran at an average of {tokens_generated/(etime-stime)/2} tokens/second.")

# Update list of available experiments
experiments_to_run = [4]  # Change this to run the new experiment







    



