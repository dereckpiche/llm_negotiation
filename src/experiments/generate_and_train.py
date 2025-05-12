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
from models.new_local_llm import LocalLLMV2
from models.server_llm import ServerLLM
from training.train_main import *
from utils.common_imports import *
from utils.update_start_epoch import update_start_epoch

# TODO (Muqeeth): * might cause circular import errors. Check with Dereck what methods we should actually import
compute_logger = logging.getLogger("compute_logger")


def generate_and_train(cfg, base_seed):
    """
    Executes a negotiation cycle for the Deal or No Deal (DoND) game.
    """
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

    # Initialize models
    models = init_models(cfg, base_seed=base_seed, output_directory=output_directory)

    for iteration in range(
        cfg["experiment"]["start_epoch"], cfg["experiment"]["nb_epochs"]
    ):
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

        logging_start_time = time.time()

        # Process raw data into training data using the specified functions for each agent
        for agent_name in agent_names:
            # Create training data directory
            training_data_path = os.path.join(it_folder, agent_name, "training")
            os.makedirs(training_data_path, exist_ok=True)

            # Get the raw data path
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

        logging_end_time = time.time()

        # Train models
        training_start_time = time.time()
        train_output_dict = {}
        for model_name, model in models.items():
            if hasattr(model, "adapter_paths"):
                for adapter_name in model.adapter_paths.keys():
                    policy_id = f"{model_name}/{adapter_name}"
                    model.prepare_adapter_train(adapter_name)

                    data_paths = []
                    for agent in agents.values():
                        if agent.policy_id == policy_id:
                            agent_export_path = os.path.join(
                                it_folder, agent.agent_name, "training"
                            )
                            data_paths.append(agent_export_path)

                    if data_paths:
                        adapter_args = cfg["training"][model_name]["adapters"][
                            adapter_name
                        ]
                        train_output = train_main(
                            hf_model=model,
                            paths=data_paths,
                            train_func=adapter_args["train_func"],
                            train_func_args=adapter_args["train_func_args"],
                            train_data_args=adapter_args["train_data_args"],
                            output_path=it_folder,
                        )
                        train_output_dict[policy_id] = train_output

        training_end_time = time.time()

        initial_logging_time = logging_end_time - logging_start_time
        logging_start_time = time.time()

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

        # TODO: Moving it here is better since we can plot for every k steps to speedup training
        for agent in agents.values():
            agent_name = agent.agent_name
            # Update agent statistics
            agent_stats_folder = os.path.join(
                output_directory, "statistics", agent_name
            )
            os.makedirs(agent_stats_folder, exist_ok=True)
            agent_stats_file = os.path.join(
                agent_stats_folder, f"{agent_name}_stats.jsonl"
            )

            update_agent_statistics(
                input_path=os.path.join(it_folder, agent_name, "statistics"),
                output_file=agent_stats_file,
            )

            with open(agent_stats_file, "r") as f:
                agent_stats = json.load(f)

            if agent.policy_id in train_output_dict:
                train_output = train_output_dict[agent.policy_id]
                for key in train_output:
                    if key in agent_stats:
                        agent_stats[key].append(train_output[key])
                    else:
                        agent_stats[key] = [train_output[key]]

            with open(agent_stats_file, "w") as f:
                json.dump(agent_stats, f, indent=4)

        logging_end_time = time.time()

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


def init_models(cfg, base_seed, output_directory):
    # TODO: just do a globals[] call
    models = {}
    for model_name in cfg["models"].keys():
        if cfg["models"][model_name]["class"] == "local_llm":
            models[model_name] = LocalLLM(
                **cfg["models"][model_name]["init_args"],
                base_seed=base_seed * cfg["experiment"]["nb_epochs"],
                output_directory=output_directory,
            )
        elif cfg["models"][model_name]["class"] == "local_llm_v2":
            models[model_name] = LocalLLMV2(
                **cfg["models"][model_name]["init_args"],
                base_seed=base_seed * cfg["experiment"]["nb_epochs"],
                output_directory=output_directory,
            )
        elif cfg["models"][model_name]["class"] == "dummy_local_llm":
            models[model_name] = DummyLocalLLM(**cfg["models"][model_name]["init_args"])
        elif cfg["models"][model_name]["class"] == "server_llm":
            models[model_name] = ServerLLM(**cfg["models"][model_name]["init_args"])
        else:
            raise ValueError(
                f"Model class {cfg['models'][model_name]['class']} not found."
            )
    return models


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
        if group_size == 0 or i % group_size == 0:
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


def get_stochastic_game_lengths(
    max_length, nb_games, continuation_prob, same_length_batch=False
):
    """
    Generates stochastic game lengths based on a geometric distribution.

    Args:
        max_length (int): The maximum length a game can have.
        nb_games (int): The number of games to generate lengths for.
        continuation_prob (float): The probability of the game continuing after each round.
        same_length_batch (bool): If True, all games will have the same length.

    Returns:
        Array: An array of game lengths.
    """
    if continuation_prob == 1:
        return [max_length] * nb_games
    if same_length_batch:
        length = np.random.geometric(1 - continuation_prob, 1)
        game_lengths = np.repeat(length, nb_games)
    else:
        game_lengths = np.random.geometric(1 - continuation_prob, nb_games)

    game_lengths = np.where(game_lengths > max_length, max_length, game_lengths)
    return game_lengths.tolist()
