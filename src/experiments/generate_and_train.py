import hydra
import os
import logging
import time
from utils.common_imports import *

# Local imports
from models.local_llm import LocalLLM
from environments.dond.dond_agent import DondAgent
from environments.dond.dond_game import DondEnv
from environments.dond.dond_training_data_funcs import *
from models.dummy_local_llm import DummyLocalLLM
from models.server_llm import ServerLLM
from utils.log_statistics import *
from utils.log_statistics import update_agent_statistics, generate_agent_stats_plots
from utils.update_start_epoch import update_start_epoch
from training.train_main import *
from generation.run_games import run_batched_matches
import torch
import numpy as np
import random
import pickle
# from environments.ipd.ipd_game import IPDEnv
# from environments.ipd.ipd_agent import IPDAgent
# from environments.dond.dond_game import DondEnv
# from environments.dond.dond_agent import DondAgent
# from environments.ipd.ipd_log_funcs import *
# from environments.dond.dond_log_funcs import *
from environments.environment_imports import *

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

    # Initialize models
    models = init_models(cfg, base_seed=base_seed, output_directory=output_directory)

    update_start_epoch(cfg=cfg, output_directory=output_directory)
    print("Start iteration: ", cfg["experiment"]["start_epoch"])

    random.seed(base_seed)  # Python random
    np.random.seed(base_seed)  # NumPy
    torch.manual_seed(base_seed)  # PyTorch (CPU)
    torch.cuda.manual_seed(base_seed)  # PyTorch (GPU)
    torch.cuda.manual_seed_all(base_seed)  # If using multi-GPU

    random_state_dir = f'{output_directory}/random_state.pkl'
    # Load saved states
    if os.path.exists(random_state_dir):
        with open(random_state_dir, "rb") as f:
            random_state_dict = pickle.load(f)
        random.setstate(random_state_dict["python"])
        np.random.set_state(random_state_dict["numpy"])
        torch.set_rng_state(random_state_dict["torch"])
        torch.cuda.set_rng_state_all(random_state_dict["torch_cuda"])

    for iteration in range(cfg["experiment"]["start_epoch"], cfg["experiment"]["nb_epochs"]):

        iteration_start_time = time.time()

        it_folder = os.path.join(output_directory, f"iteration_{iteration:03}")
        os.makedirs(it_folder, exist_ok=True)

        generation_start_time = time.time()

        # Create independent matches based on the number in config
        matches = []
        nb_matches = cfg["experiment"]["nb_matches_per_iteration"]
        for i in range(nb_matches):
            matches.append(create_blank_match(cfg, seed_offset=(iteration * nb_matches) + i))
        agents = matches[0]["agents"]
        agent_names = agents.keys()

        # Run matches to collect raw conversation data
        run_batched_matches(
            export_path=it_folder,
            matches=matches,
            seed_offset=iteration,
            models=models,
            **cfg['matches']['run_batched_matches_args']
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
                    **training_data_func_args
                )

            # Update agent statistics
            agent_stats_folder = os.path.join(output_directory, "statistics", agent_name)
            os.makedirs(agent_stats_folder, exist_ok=True)
            agent_stats_file = os.path.join(agent_stats_folder, f"{agent_name}_stats.jsonl")

            update_agent_statistics(
                input_path=os.path.join(it_folder, agent_name, "statistics"),
                output_file=agent_stats_file
            )

            generate_agent_stats_plots(
                global_stats_path=agent_stats_file,
                matplotlib_log_dir=os.path.join(agent_stats_folder, "matplotlib"),
                tensorboard_log_dir=os.path.join(agent_stats_folder, "tensorboard"),
                wandb_log_dir=os.path.join(agent_stats_folder, "wandb"),
            )

        logging_end_time = time.time()

        # Train models
        training_start_time = time.time()

        for model_name, model in models.items():
            if hasattr(model, 'adapters'):
                for adapter_name in model.adapters.keys():
                    policy_id = f"{model_name}/{adapter_name}"
                    model.prepare_adapter_train(adapter_name)

                    data_paths = []
                    for agent in agents.values():
                        if agent.policy_id == policy_id:
                            agent_export_path = os.path.join(it_folder, agent.agent_name, "training")
                            data_paths.append(agent_export_path)

                    if data_paths:
                        train_func_args = cfg["training"][model_name]["adapters"][adapter_name]["train_func_args"]
                        train_main(
                            hf_model=model,
                            paths=data_paths,
                            train_func=cfg["training"][model_name]["adapters"][adapter_name]["train_func"],
                            train_func_args=train_func_args,
                            output_path=it_folder
                        )

        training_end_time = time.time()

        iteration_end_time = time.time()

        # Timing calculations
        iteration_duration = iteration_end_time - iteration_start_time
        generation_duration = generation_end_time - generation_start_time
        logging_duration = logging_end_time - logging_start_time
        training_duration = training_end_time - training_start_time

        generation_percentage = (generation_duration / iteration_duration) * 100
        logging_percentage = (logging_duration / iteration_duration) * 100
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
            f"({generation_percentage:.2f}% Gen, {logging_percentage:.2f}% Log, {training_percentage:.2f}% Train). "
            f"Generation: {format_time(generation_duration)}, "
            f"Logging: {format_time(logging_duration)}, "
            f"Training: {format_time(training_duration)}. "
            f"Estimated remaining time: {format_time(estimated_remaining_time)}. "
            f"Estimated total time: {format_time(estimated_total_time)}. "
            f"Time estimates for 10 more iterations: {format_time(time_est_10)}, "
            f"100 more iterations: {format_time(time_est_100)}, "
            f"500 more iterations: {format_time(time_est_500)}."
        )

        # Save Python random state
        python_random_state = random.getstate()

        # Save NumPy random state
        numpy_random_state = np.random.get_state()

        # Save PyTorch random state
        torch_random_state = torch.get_rng_state()
        torch_cuda_random_state = torch.cuda.get_rng_state_all()  # For all GPUs

        # Store in a dictionary (or save to a file)
        random_state_dict = {
            "python": python_random_state,
            "numpy": numpy_random_state,
            "torch": torch_random_state,
            "torch_cuda": torch_cuda_random_state,
        }

        with open(random_state_dir, "wb") as f:
            pickle.dump(random_state_dict, f)

        print("Saved random states!")

    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    compute_logger.info(f"Total time taken for the entire run: {format_time(total_duration)}")



def init_models(cfg, base_seed, output_directory):
    models = {}
    for model_name in cfg["models"].keys():
        if cfg["models"][model_name]["class"] == "local_llm":
            models[model_name] = LocalLLM(**cfg["models"][model_name]["init_args"], base_seed=base_seed * cfg["experiment"]["nb_epochs"],
                                         output_directory=output_directory)
        elif cfg["models"][model_name]["class"] == "dummy_local_llm":
            models[model_name] = DummyLocalLLM(**cfg["models"][model_name]["init_args"])
        elif cfg["models"][model_name]["class"] == "server_llm":
            models[model_name] = ServerLLM(**cfg["models"][model_name]["init_args"])
        else:
            raise ValueError(f"Model class {cfg['models'][model_name]['class']} not found.")
    return models


def create_blank_match(cfg, seed_offset=0):
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
    # import pdb; pdb.set_trace()
    for agent_name in cfg["matches"]["agents"].keys():
        agents[agent_name] = AgentClass(
            **cfg["matches"]["agents"][agent_name]["kwargs"]
        )

    # Get environment class from config
    env_class_name = cfg["matches"]["env_class"]
    EnvClass = globals()[env_class_name]

    # Build a fresh copy of game args to safely update random setup parameters
    env_kwargs = dict(cfg["matches"]["env_kwargs"])

    # Handle random setup kwargs if they exist in the config
    if "random_setup_kwargs" in env_kwargs:
        setup_kwargs = env_kwargs.get("random_setup_kwargs", {})
        setup_kwargs = dict(setup_kwargs)  # shallow copy
        # Put back the cleaned up random_setup_kwargs
        env_kwargs["random_setup_kwargs"] = setup_kwargs

    # Create match with instantiated environment and agents
    env = EnvClass(
        random_seed=seed_offset,  # Pass the unique seed here
        **env_kwargs
    )


    # Add the logging function and args to the match dictionary
    match = {
        'env': env,
        'agents': agents,
        'log_func': globals()[cfg['matches']['log_func']],
        'log_func_args': cfg['matches']['log_func_args']
    }

    return match

def format_time(seconds):
    if seconds >= 3600:
        return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m {int(seconds % 60)}s"
    elif seconds >= 60:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    else:
        return f"{int(seconds)}s"