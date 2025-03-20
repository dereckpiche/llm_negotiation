import torch
from vllm import *
import time

# see https://docs.vllm.ai/en/latest/serving/engine_args.html
vllm_params = {
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "dtype": torch.bfloat16,
    "enable_lora": True,
    "max_model_len": 13000,
    "gpu_memory_utilization": 0.45,
    "device": torch.device("cuda:0"),
    "enforce_eager": False,  # Default False
    "block_size": 128,  # Default: 128
    "enable_prefix_caching": True,
    "disable_custom_all_reduce": False,  # Default False
    "seed": 42,
    "enable_chunked_prefill": False  # Default False
}


llm_agent1 = LLM(**vllm_params)

llm_agent2 = LLM(**vllm_params)

              

experiments_to_run = [2]

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


if 2 in experiments_to_run:
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

    # This took 2.09 minutes to run vllm WITH "enable_prefix_caching": True, total_games = 32, parallel_games = 32, nb_rounds = 16, nb_turns=2, intro_prompt_length=800, intermediary_length=200, response_lengths=[100, 20]







        
    


