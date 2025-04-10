import torch
from vllm import *
import time

# see https://docs.vllm.ai/en/latest/serving/engine_args.html
vllm_params = {
    "model": "meta-llama/Llama-3.1-8B-Instruct",
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
    "enable_chunked_prefill": False  # Default False
}


experiments_to_run = [4]

##############################################################################
# Experiment 3: single vLLM instance
##############################################################################

if 1 in experiments_to_run:
    for _ in range(5): print(150*"=")
    total_games = 32
    parallel_games = 5
    tokens_to_gen = 10000
    initial_prompts = [ str(n) + " Hi, " for n in range(total_games) ]
    print("Experiment 1 running")
    stime = time.time()
    for i in range(0, int(total_games/parallel_games)):
        prompts = initial_prompts[i*parallel_games: (i+1)*parallel_games]
        print(prompts)
        output1 = llm_agent1.generate(prompts, sampling_params=SamplingParams(skip_special_tokens=True, max_tokens=tokens_to_gen))
    etime = time.time()
    print(f"Experiment 1 took {(etime-stime)/60} minutes.")
    # This took 10 minutes with vllm default, total_games=32, parallel_games=15, tokens_to_gen=10000

##############################################################################
# Experiment 2: two instances vLLM
##############################################################################

if 2 in experiments_to_run:
    llm_agent1 = LLM(**vllm_params)
    llm_agent2 = LLM(**vllm_params)
    """
    This experiment runs simulated games with back and forths between two vllm instances 
    running on a single GPU and prints the time taken in minutes.
    """
    for _ in range(5): print(150*"=")
    print("Experiment 2: Conversation Speeds, running...")
    total_games = 32
    parallel_games = 32
    nb_rounds=16
    nb_turns=2
    intro_prompt_length=800
    intermediary_length=200
    response_lengths=[100, 20]

    full_contexts_1 = [str(i) + intro_prompt_length*"0" for i in range(total_games)]
    full_contexts_2 = [str(i) + intro_prompt_length*"1" for i in range(total_games)]

    print("Experiment 2 running")
    stime = time.time()
    for i in range(0, int(total_games/parallel_games)):

        contexts_1 = full_contexts_1[i*parallel_games: (i+1)*parallel_games]

        contexts_2 = full_contexts_2[i*parallel_games: (i+1)*parallel_games]

        for j in range(nb_rounds):

            for k in range(nb_turns):

                responses_1 = llm_agent1.generate(contexts_1, sampling_params=SamplingParams(skip_special_tokens=True, max_tokens=response_lengths[k]))
                responses_1 = [r.outputs[0].text for r in responses_1]
                contexts_1 = [c+r for c,r in zip(contexts_1, responses_1)]
                contexts_1 = [c+intermediary_length*"0" for c in contexts_1]

                responses_2 = llm_agent2.generate(contexts_2, sampling_params=SamplingParams(skip_special_tokens=True, max_tokens=response_lengths[k]))
                responses_2 = [r.outputs[0].text for r in responses_2]
                contexts_2 = [c+r for c,r in zip(contexts_2, responses_2)]
                contexts_2 = [c+intermediary_length*"1" for c in contexts_2]

    etime = time.time()
    print(f"Experiment 2 took {(etime-stime)/60} minutes.")
    import math
    tokens_generated = total_games * nb_rounds * math.sum(response_lengths) 
    print(f"Each model ran at an average of {tokens_generated/(etime-stime)/2} tokens/second.")

    # This took  7.545 minutes to run vllm default, total_games = 32, parallel_games = 32, nb_rounds = 16, nb_turns=2, intro_prompt_length=800, intermediary_length=100, response_lengths=[100, 100]

    # This took 5.92 minutes to run vllm default, total_games = 32, parallel_games = 32, nb_rounds = 16, nb_turns=2, intro_prompt_length=800, intermediary_length=100, response_lengths=[120, 30]

    # This took 6.22 minutes to run vllm default, total_games = 32, parallel_games = 32, nb_rounds = 16, nb_turns=2, intro_prompt_length=800, intermediary_length=200, response_lengths=[100, 20]

    # This took 3.46 minutes to run vllm default, total_games = 16, parallel_games = 16, nb_rounds = 16, nb_turns=2, intro_prompt_length=800, intermediary_length=200, response_lengths=[100, 20]

    # This took 1.40 minutes to run vllm WITH "enable_prefix_caching": True, total_games = 16, parallel_games = 16, nb_rounds = 16, nb_turns=2, intro_prompt_length=800, intermediary_length=200, response_lengths=[100, 20]

    # This took 2.09 minutes to run vllm WITH "enable_prefix_caching": True, total_games = 32, parallel_games = 32, nb_rounds = 16, nb_turns=2, intro_prompt_length=800, intermediary_length=200, response_lengths=[100, 20]ef
    


##############################################################################
# Experiment 3: HF adapters
##############################################################################

if 3 in experiments_to_run:
    print("=" * 150)
    print("Experiment 3: Conversation Speeds using single HF instance with 2 LoRA adapters, running...")

    # Import necessary libraries
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    import copy

    # Load the Hugging Face base model and tokenizer
    hf_model_name = "meta-llama/Llama-3.1-8B-Instruct"
    base_model = AutoModelForCausalLM.from_pretrained(hf_model_name, torch_dtype=torch.bfloat16).to(vllm_params["device"])
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Define the LoRA configuration (adjust parameters as needed)
    lora_config = LoraConfig(
         r=8,
         lora_alpha=32,
         lora_dropout=0.1,
         target_modules=["q_proj", "v_proj"],  # adjust these target modules based on your model architecture
         bias="none",
         task_type="CAUSAL_LM"
    )
    # Create adapter 1 on the base model
    model_adapter1 = get_peft_model(base_model, lora_config)
    # Create adapter 2 on a deepcopy of the base model (to ensure adapter independence)
    model_adapter2 = get_peft_model(base_model, lora_config)

    # Set up experiment parameters (similar to Experiment 2)
    total_games = 32
    parallel_games = 32
    nb_rounds = 16
    nb_turns = 2
    intro_prompt_length = 800
    intermediary_length = 200
    response_lengths = [100, 20]

    # Create initial contexts for the two simulated agents
    full_contexts_1 = [str(i) + "0" * intro_prompt_length for i in range(total_games)]
    full_contexts_2 = [str(i) + "1" * intro_prompt_length for i in range(total_games)]

    print("Experiment 3 running")
    stime = time.time()
    for i in range(0, int(total_games / parallel_games)):
        contexts_1 = full_contexts_1[i * parallel_games: (i + 1) * parallel_games]
        contexts_2 = full_contexts_2[i * parallel_games: (i + 1) * parallel_games]
        for j in range(nb_rounds):
            for k in range(nb_turns):
                # Generate responses for agent 1 using adapter1
                inputs_1 = tokenizer(contexts_1, return_tensors="pt", padding=True).to(base_model.device)
                with torch.no_grad():
                    outputs_1 = model_adapter1.generate(**inputs_1,
                                                          max_new_tokens=response_lengths[k],
                                                          do_sample=True)
                responses_1 = tokenizer.batch_decode(outputs_1, skip_special_tokens=True)
                contexts_1 = [c + r for c, r in zip(contexts_1, responses_1)]
                contexts_1 = [c + "0" * intermediary_length for c in contexts_1]

                # Generate responses for agent 2 using adapter2
                inputs_2 = tokenizer(contexts_2, return_tensors="pt", padding=True).to(base_model.device)
                with torch.no_grad():
                    outputs_2 = model_adapter2.generate(**inputs_2,
                                                          max_new_tokens=response_lengths[k],
                                                          do_sample=True)
                responses_2 = tokenizer.batch_decode(outputs_2, skip_special_tokens=True)
                contexts_2 = [c + r for c, r in zip(contexts_2, responses_2)]
                contexts_2 = [c + "1" * intermediary_length for c in contexts_2]
    etime = time.time()
    print(f"Experiment 3 took {(etime - stime) / 60} minutes.")
    import math
    tokens_generated = total_games * nb_rounds * sum(response_lengths)
    print(f"Each adapter ran at an average of {tokens_generated / (etime - stime) / 2} tokens/second.")

##############################################################################
# Experiment 4: Two vLLM instances on different GPUs using multiprocessing
##############################################################################

if 4 in experiments_to_run:
    print("=" * 150)
    print("Experiment 4: Conversation Speeds with two vLLM instances on different GPUs, running...")
    
    import multiprocessing
    import os
    import numpy as np
    
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
                
            task_id, contexts, max_tokens = task
            
            # Generate responses
            responses = llm_agent.generate(contexts, sampling_params=SamplingParams(
                skip_special_tokens=True, 
                max_tokens=max_tokens
            ))
            responses = [r.outputs[0].text for r in responses]
            
            # Send results back
            queue_out.put((task_id, responses))
    
    # Set up experiment parameters
    total_games = 32
    parallel_games = 32
    nb_rounds = 16
    nb_turns = 2
    intro_prompt_length = 800
    intermediary_length = 200
    response_lengths = [100, 20]
    
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
    
    print("Experiment 4 running")
    stime = time.time()
    
    for i in range(0, int(total_games/parallel_games)):
        contexts_1 = full_contexts_1[i*parallel_games: (i+1)*parallel_games]
        contexts_2 = full_contexts_2[i*parallel_games: (i+1)*parallel_games]
        
        for j in range(nb_rounds):
            for k in range(nb_turns):
                # Send requests to both processes
                q1_in.put((f"r{j}t{k}", contexts_1, response_lengths[k]))
                q2_in.put((f"r{j}t{k}", contexts_2, response_lengths[k]))
                
                # Get responses from both processes
                _, responses_1 = q1_out.get()
                _, responses_2 = q2_out.get()
                
                # Update contexts
                contexts_1 = [c+r for c,r in zip(contexts_1, responses_1)]
                contexts_1 = [c+intermediary_length*"0" for c in contexts_1]
                
                contexts_2 = [c+r for c,r in zip(contexts_2, responses_2)]
                contexts_2 = [c+intermediary_length*"1" for c in contexts_2]
    
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







        
    


