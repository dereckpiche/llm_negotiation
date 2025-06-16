"""
This file contains the code to generate and train the models.
"""
import copy
import logging
import os
import pickle
import random
import time
from typing import Dict, List, Any

import hydra
import numpy as np
import torch

# Local imports
from environments.env_imports import *
from generation.run_games import run_batched_matches
from models.dummy_local_llm import DummyLocalLLM
from models.local_llm import LocalLLM
from models.lean_local_llm import LeanLocalLLM
from models.server_llm import ServerLLM
from models.critic_wrapper import ScalarCritic
from training.reinforce_trainer import *
from training.reinforce_trainer_config import *
from training.reinforce_trainer_tally import *
from utils.common_imports import *
from utils.update_start_epoch import update_start_epoch
from utils.get_stochastic_game_lengths import get_stochastic_game_lengths
from utils.dict_get_path import get_at

compute_logger = logging.getLogger("compute_logger")



def create_markov_games(cfg: Dict[str, Any], env_rng: np.random.Generator, iteration: int) -> tuple[List[Dict[str, Any]], np.random.Generator]:
    """
    Creates a list of Markov games for training.
    
    Args:
        cfg: Configuration dictionary
        env_rng: Random number generator for environment
        iteration: Current iteration number
        
    Returns:
        Tuple containing:
        - List of Markov games
        - Updated random number generator
    """
    markov_games = []
    nb_games = cfg["experiment"]["nb_matches_per_iteration"]
    game_lengths = get_stochastic_game_lengths(
        max_length=cfg["matches"]["max_length"],
        nb_games=nb_games,
        continuation_prob=cfg["matches"]["continuation_prob"],
        same_length_batch=cfg["matches"]["same_length_batch"],
    )
    group_size = cfg["matches"]["nb_matches_with_same_roundwise_utilities"]
    for i in range(nb_games):
        if group_size is not None and (i % group_size == 0):
            env_rng = np.random.default_rng(env_rng.integers(0, 1e9))
            # Create Markov Game
            agents = {}
            agent_class_name = cfg["matches"]["agent_class"]
            AgentClass = globals()[agent_class_name]
            for agent_name in cfg["matches"]["agents"].keys():
                agents[agent_name] = AgentClass(
                    **cfg["matches"]["agents"][agent_name]["kwargs"]
                )
            env_class_name = cfg["matches"]["env_class"]
            EnvClass = globals()[env_class_name]
            env_kwargs = dict(cfg["matches"]["env_kwargs"])
            rng = copy.deepcopy(env_rng)
            game_id = i
            game_length = game_lengths[i]
            group_id = i // group_size if group_size else -1
            env = EnvClass(
                rng=rng,
                game_id=game_id,
                group_id=group_id,
                rounds_per_game=game_length,
                **env_kwargs,
            )
            markov_game = {
                "env": env,
                "agents": agents,
                "log_func": globals()[cfg["matches"]["log_func"]],
            }
            markov_games.append(markov_game)
    return markov_games, env_rng
    

def generate_and_train(cfg: dict, base_seed: int) -> None:
    """
    Main function to generate training data and train models.
    
    Args:
        cfg: Configuration dictionary
        base_seed: Base seed for random number generation
    """

    # -----------------------------------------------------------------
    # Initialize Random States (+ check if resume run) 
    # -----------------------------------------------------------------

    total_start_time = time.time()
    # Get Hydra's runtime output directory which includes date and config info.
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_directory = f"{hydra_cfg['runtime']['output_dir']}/seed_{base_seed}"
    os.makedirs(output_directory, exist_ok=True)

    update_start_epoch(cfg=cfg, output_directory=output_directory)
    print("Start iteration: ", cfg["experiment"]["start_epoch"])

    random.seed(base_seed)  # Python random
    np.random.seed(base_seed)  # NumPy
    torch.manual_seed(base_seed)  # PyTorch (CPU)
    torch.cuda.manual_seed(base_seed)  # PyTorch (GPU)
    torch.cuda.manual_seed_all(base_seed)  # If using multi-GPU

    env_rng = np.random.default_rng(base_seed)

    random_state_dir = f"{output_directory}/random_state.pkl"
    # Load saved states
    if os.path.exists(random_state_dir):
        with open(random_state_dir, "rb") as f:
            random_state_dict = pickle.load(f)
        print(f"Loaded random states from {random_state_dir}")
        random.setstate(random_state_dict["python"])
        np.random.set_state(random_state_dict["numpy"])
        torch.set_rng_state(random_state_dict["torch"])
        torch.cuda.set_rng_state_all(random_state_dict["torch_cuda"])

    # -----------------------------------------------------------------
    # Initialize models, critics, optimizers, trainers
    # -----------------------------------------------------------------

    # Models & adapters
    shared_llms = {}
    for shared_llm_id, model_config in cfg["models"].items():
        model_class = globals()[model_config["class"]]
        shared_llms[shared_llm_id] = model_class(
            **model_config["init_args"],
            output_directory=output_directory,
        )

    trainable_modules = {}
    for shared_llm_id, shared_llm in shared_llms.items():
        trainable_modules[shared_llm_id] = shared_llm.get_adapter_pointers()

    # Critics
    for critic_id, critic_config in cfg["critics"].items():
        pointer = critic_config["pointer"]
        base = get_at(trainable_modules, pointer)
        trainable_modules[critic_id] = ScalarCritic(base)

    # Optimizers
    optimizers = {}
    for optimizer_id, optimizer_config in cfg["optimizers"].items():
        pointer = optimizer_config["pointer"]
        base = get_at(trainable_modules, pointer)
        optimizer_class = eval(optimizer_config["optimizer_class"])
        init_args = optimizer_config["init_args"]
        optimizers[optimizer_id] = optimizer_class(base.parameters(), **init_args)

    # Trainers
    trainers = {}
    for trainer_id, trainer_config in cfg["trainers"].items():
        pointers = trainer_config["pointers"]
        tokenizer = shared_llms[pointers["model"][0]].tokenizer
        policy_model = get_at(trainable_modules, pointers["model"])
        policy_optimizer = get_at(optimizers, pointers["optimizer"])
        if pointers.get("critic", True): 
            critic_model = get_at(
                trainable_modules, pointers["critic"])
        else: critic_model = None
        if pointers.get("critic_optimizer", True): 
            critic_optimizer = get_at(
                optimizers, pointers["critic_optimizer"])
        else: critic_optimizer = None
        trainer = ReinforceTrainerWRS(
                    model=policy_model,
                    optimizer=policy_optimizer,
                    tokenizer=tokenizer,
                    lr_scheduler=None, #TODO add
                    critic=critic_model,
                    critic_optimizer=critic_optimizer,
                    critic_lr_scheduler=None, # #TODO add
                    config=RtConfig(
                        **trainer_config["init_args"],
                        logging_path=None,
                    ),
                )
        trainers[trainer_id] = trainer

    import pdb; pdb.set_trace()



    for iteration in range(cfg["experiment"]["start_epoch"], cfg["experiment"]["nb_epochs"]):

        # -----------------------------------------------------------------
        # Create Training Rollouts 
        # -----------------------------------------------------------------

        # TODO: prepare base models for evaluation
        for shared_llm in shared_llms.values():
            shared_llm.toggle_eval_mode()

        # Create a new RNG instance by splitting the current one (simulates RNG splitting)
        env_rng = np.random.default_rng(env_rng.integers(0, 1e9))
        iteration_start_time = time.time()
        it_folder = os.path.join(output_directory, f"iteration_{iteration:03}")
        os.makedirs(it_folder, exist_ok=True)
        generation_start_time = time.time()

        # Gen. rollouts
        matches, env_rng = create_markov_games(cfg, env_rng, iteration)
        agents = matches[0]["agents"]
        agent_names = agents.keys()
        run_batched_matches(
            export_path=it_folder,
            matches=matches,
            seed_offset=iteration,
            models=shared_llms,
            **cfg["matches"]["run_batched_matches_args"],
        )

        generation_end_time = time.time()

        # Process raw data into training data using the specified functions for each agent
        for agent_name in agent_names:
            training_data_path = os.path.join(it_folder, agent_name, "training")
            os.makedirs(training_data_path, exist_ok=True)
            raw_data_path = os.path.join(it_folder, agent_name, "raw_data")
            if agent_name in cfg["training"]["agents"]:
                agent_cfg = cfg["training"]["agents"][agent_name]
                training_data_func = agent_cfg.get("training_data_func")
                training_data_func_args = agent_cfg.get("training_data_func_args", {})
                globals()[training_data_func](
                    raw_data_folder=raw_data_path,
                    training_data_folder=training_data_path,
                    **training_data_func_args,
                )

        # -----------------------------------------------------------------
        # Train
        # -----------------------------------------------------------------
        training_start_time = time.time()


        # Get training paths for each trainer
        trainer_training_paths = {}
        for trainer_id, trainer in trainers.items():
            hf_adapter_id = cfg["trainers"][trainer_id]["pointers"]["model"]
            policy_id = f"{hf_adapter_id[0]}/{hf_adapter_id[1]}"
            training_file_paths = []
            for agent in agents.values():
                if agent.policy_id == policy_id:
                    agent_export_path = os.path.join(
                        it_folder, agent.agent_name, "training"
                    )
                    filepaths = [
                        os.path.join(agent_export_path, path)
                        for path in os.listdir(agent_export_path)
                    ]
                    training_file_paths += filepaths
            trainer_training_paths[trainer_id] = training_file_paths


        # Prepare base models for training 
        for shared_llm in shared_llms.values():
            shared_llm.toggle_training_mode()


        # Training Phase 1: set training data, opp shaping info transfers
        shaping_info_sets = []
        for trainer_id, trainer in trainers.items():
            trainer.set_training_data(
                paths = trainer_training_paths[trainer_id]
            )
            info = trainer.send_shaping_info_to_opponents()
            shaping_info_sets.append(info)

        # Training Phase 2: use opp shaping infos, apply reinforce
        trainer_items = list(trainers.items())
        for i, (trainer_id, trainer) in enumerate(trainer_items):
            info = shaping_info_sets[1-i] 
            trainer.use_shaping_info_to_opponents(
                opponents_info=info
            )
            train_log_out_path = os.path.join(
                    it_folder, 
                    "training_metrics", 
                    trainer_id
                )
            trainer.config.logging_path = train_log_out_path
            trainer.train()
            trainer.export_training_metrics()

        # Export all HF adapters weights (needed for vLLM inference)
        for shared_llm in shared_llms.values(): shared_llm.export_adapters()

        # TODO: export trainers! (trainers export critics and otimizers)
        for shared_llm in shared_llms.values(): shared_llm.export_adapters()


        training_end_time = time.time()


        # Checkpoint all adapters
        checkpoint_frequency = cfg["experiment"]["checkpoint_every_n_iterations"]
        if (
            checkpoint_frequency != -1
            and iteration % checkpoint_frequency == 0
            and iteration != 0
        ):
            for shared_llm_id, shared_llm in shared_llms.items():
                if hasattr(shared_llm, "adapter_paths"):
                    shared_llm.checkpoint_all_adapters(
                        checkpoint_indicator=f"iter_{iteration}"
                    )


        iteration_end_time = time.time()

        # Timing calculations
        iteration_duration = iteration_end_time - iteration_start_time
        generation_duration = generation_end_time - generation_start_time
        training_duration = training_end_time - training_start_time

        generation_percentage = (generation_duration / iteration_duration) * 100
        training_percentage = (training_duration / iteration_duration) * 100

        elapsed_time = iteration_end_time - total_start_time
        estimated_total_time = iteration_duration * cfg["experiment"]["nb_epochs"]
        estimated_remaining_time = estimated_total_time - elapsed_time

        time_per_iteration = iteration_duration
        time_est_10 = time_per_iteration * 10
        time_est_100 = time_per_iteration * 100
        time_est_500 = time_per_iteration * 500

        compute_logger.info(
            f"Iteration {iteration + 1} took {format_time(iteration_duration)} "
            f"({generation_percentage:.2f}% Gen, {training_percentage:.2f}% Train). "
            f"Generation: {format_time(generation_duration)}, "
            f"Training: {format_time(training_duration)}. "
            f"Estimated remaining time: {format_time(estimated_remaining_time)}. "
            f"Estimated total time: {format_time(estimated_total_time)}. "
            f"Time estimates for 10 more iterations: {format_time(time_est_10)}, "
            f"100 more iterations: {format_time(time_est_100)}, "
            f"500 more iterations: {format_time(time_est_500)}."
        )

        python_random_state = random.getstate()
        numpy_random_state = np.random.get_state()
        torch_random_state = torch.get_rng_state()
        torch_cuda_random_state = torch.cuda.get_rng_state_all()

        # Store in a dictionary (or save to a file)
        random_state_dict = {
            "python": python_random_state,
            "numpy": numpy_random_state,
            "torch": torch_random_state,
            "torch_cuda": torch_cuda_random_state,
        }

        with open(random_state_dir, "wb") as f:
            pickle.dump(random_state_dict, f)

    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    compute_logger.info(
        f"Total time taken for the entire run: {format_time(total_duration)}"
    )




def format_time(seconds):
    if seconds >= 3600:
        return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m {int(seconds % 60)}s"
    elif seconds >= 60:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    else:
        return f"{int(seconds)}s"



