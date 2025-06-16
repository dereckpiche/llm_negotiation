"""
This file contains the code to generate and train the models.
"""
import copy
import logging
import os
import pickle
import random
import time

import hydra
import numpy as np
import torch

# Local imports
from environments.env_imports import *
from generation.run_games import run_batched_matches
from models.dummy_local_llm import DummyLocalLLM
from models.local_llm import LocalLLM
from src.models.local_llm import LocalLLMV2
from models.server_llm import ServerLLM
from training.reinforce_trainer import *
from training.reinforce_trainer_config import *
from training.reinforce_trainer_tally import *
from utils.common_imports import *
from utils.update_start_epoch import update_start_epoch
from utils.get_stochastic_game_lengths import get_stochastic_game_lengths
from utils.dict_get_path import get_at
compute_logger = logging.getLogger("compute_logger")



def create_markov_games(cfg: dict, env_rng, iteration: int):
    """
    (...)
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
    

def generate_and_train(cfg, base_seed):
    """
    (...)
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
    models = {}
    for model_id, model_config in cfg["models"].items():
        model_class = globals()[model_config["class"]]
        models[model_id] = model_class(
            **model_config["init_args"],
            base_seed=base_seed * cfg["experiment"]["nb_epochs"],
            output_directory=output_directory,
        )
    hf_adapter_pointers = {}
    for model_id, model in models:
        hf_adapter_pointers[model_id] = model.get_adapter_pointers()

    # Critics
    critics = {}
    from models.critic_wrapper import ScalarCritic
    for critic_id, critic_config in cfg["configs"].items():
        pointer = critic_config[pointer]
        base = get_at(hf_adapter_pointers, pointer)
        critics[critic_id] = ScalarCritic(base)

    # Optimizers
    optimizers = {}
    for optimizer_id, optimizer_config in cfg["optimizers"].items():
        pointer = critic_config[pointer]
        base = get_at(hf_adapter_pointers, pointer)
        optimizer_class = globals()[optimizer_config["optimizer_class"]]
        init_args = optimizer_config["init_args"]
        optimizers[optimizer_id] = optimizer_class(base, **init_args)

    # Trainers
    trainers = {}
    for trainer_id, trainer_config in cfg["trainers"].items():
        pointers = critic_config[pointers]
        model = get_at(hf_adapter_pointers, pointers["model"])
        optimizer = get_at(hf_adapter_pointers, pointers["model"])
        if pointers.get("critic", True): critic = get_at(hf_adapter_pointers, pointers["critic"])
        else: critic = None
        if pointers.get("critic_optimizer", True): critic_optimizer = get_at(hf_adapter_pointers, pointers["critic_optimizer"])
        else: critic_optimizer = None
        trainer = ReinforceTrainerWRS(
                    model=model,
                    optimizer=optimizer,
                    tokenizer=model.tokenizer,
                    lr_scheduler=None,
                    critic=critic,
                    critic_optimizer=critic_optimizer,
                    config=RtConfig(
                        **trainer_config["init_args"],
                        logging_path=train_output_path,
                    ),
                )
        trainers[trainer_id] = trainer



    for iteration in range(
        cfg["experiment"]["start_epoch"], cfg["experiment"]["nb_epochs"]
    ):

        # -----------------------------------------------------------------
        # Create Training Rollouts 
        # -----------------------------------------------------------------

        # TODO: prepare base models for evaluation
        for model in models.vals():
            model.toggle_eval_mode()

        # Create a new RNG instance by splitting the current one (simulates RNG splitting)
        env_rng = np.random.default_rng(env_rng.integers(0, 1e9))

        iteration_start_time = time.time()

        it_folder = os.path.join(output_directory, f"iteration_{iteration:03}")
        os.makedirs(it_folder, exist_ok=True)

        generation_start_time = time.time()

        # Create independent matches based on the number in config.
        # Return modified RNG so that a different split is generated for the next iteration.
        matches, env_rng = create_matches(cfg, env_rng, iteration)

        agents = matches[0]["agents"]
        agent_names = agents.keys()

        # Run matches to collect raw conversation data
        run_batched_matches(
            export_path=it_folder,
            matches=matches,
            seed_offset=iteration,
            models=models,
            **cfg["matches"]["run_batched_matches_args"],
        )
        del matches

        generation_end_time = time.time()

        # Process raw data into training data using the specified functions for each agent
        for agent_name in agent_names:
            # Create training data directory
            training_data_path = os.path.join(it_folder, agent_name, "training")
            os.makedirs(training_data_path, exist_ok=True)
            raw_data_path = os.path.join(it_folder, agent_name, "raw_data")
            # Process the raw data using the specified training data function
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
        for model in models.vals():
            model.toggle_training_mode()


        # Set training data for each trainers
        for trainer_id, trainer in trainers.items():
            


        # Allow opponent shaping info to be shared


        
                if training_file_paths:
                    adapter_args = cfg["training"][model_name]["adapters"][
                        adapter_name
                    ]

                    train_output_path = os.path.join(
                        it_folder, "training_metrics", adapter_name
                    )

                    trainer = ReinforceTrainerWRS(
                        model=model.hf_model,
                        optimizer=model.optimizer,
                        tokenizer=model.tokenizer,
                        lr_scheduler=None,
                        config=RtConfig(
                            **adapter_args["trainer_config"],
                            logging_path=train_output_path,
                        ),
                    )
                    trainer.apply_reinforce_step_on_paths(paths=training_file_paths)
                    trainer.export_training_metrics()
                    model.export_current_adapter_and_optimizer()
                    del trainer

        training_end_time = time.time()


        # Checkpoint all adapters
        checkpoint_frequency = cfg["experiment"]["checkpoint_every_n_iterations"]
        if (
            checkpoint_frequency != -1
            and iteration % checkpoint_frequency == 0
            and iteration != 0
        ):
            for model_name, model in models.items():
                if hasattr(model, "adapter_paths"):
                    model.checkpoint_all_adapters(
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


def create_matches(cfg, env_rng, iteration):
    matches = []
    nb_matches = cfg["experiment"]["nb_matches_per_iteration"]
    game_lengths = get_stochastic_game_lengths(
        max_length=cfg["matches"]["max_length"],
        nb_games=nb_matches,
        continuation_prob=cfg["matches"]["continuation_prob"],
        same_length_batch=cfg["matches"]["same_length_batch"],
    )

    # Number of games that share the same utilities
    group_size = cfg["matches"]["nb_matches_with_same_roundwise_utilities"]

    for i in range(nb_matches):
        if group_size is not None and (i % group_size == 0):
            env_rng = np.random.default_rng(env_rng.integers(0, 1e9))

        matches.append(
            create_blank_match(
                cfg,
                # Maintain the same RNG for a group / minibatch.
                rng=copy.deepcopy(env_rng),
                game_id=i,
                game_length=game_lengths[i],
                # Minibatch / group id for which roundwise utilities are same
                group_id=i // group_size if group_size else -1,
            )
        )

    return matches, env_rng


def create_blank_match(cfg, rng, game_id=0, game_length=10, group_id=0):
    """
    Initializes a match for any game, using a functional approach to instantiate
    environment and agent classes based on configuration.

    Args:
        cfg (omegaconf.DictConfig): Configuration object containing all necessary parameters.
        seed_offset (int): An offset to uniquely adjust the seed for each match.

    Returns:
        dict: A match dictionary containing environment and agents.
    """
    agents = {}

    # Create agents using the class specified in config
    agent_class_name = cfg["matches"]["agent_class"]
    AgentClass = globals()[agent_class_name]
    for agent_name in cfg["matches"]["agents"].keys():
        agents[agent_name] = AgentClass(
            **cfg["matches"]["agents"][agent_name]["kwargs"]
        )

    # Get environment class from config
    env_class_name = cfg["matches"]["env_class"]
    EnvClass = globals()[env_class_name]
    env_kwargs = dict(cfg["matches"]["env_kwargs"])

    env = EnvClass(
        rng=rng,
        game_id=game_id,
        group_id=group_id,
        rounds_per_game=game_length,
        **env_kwargs,
    )

    # Add the logging function and args to the match dictionary
    log_func = lambda path, agent_infos, info: globals()[cfg["matches"]["log_func"]](
        path, agent_infos, info, **cfg["matches"]["log_func_args"]
    )

    match = {
        "env": env,
        "agents": agents,
        "log_func": globals()[cfg["matches"]["log_func"]],
    }

    return match


def format_time(seconds):
    if seconds >= 3600:
        return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m {int(seconds % 60)}s"
    elif seconds >= 60:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    else:
        return f"{int(seconds)}s"



