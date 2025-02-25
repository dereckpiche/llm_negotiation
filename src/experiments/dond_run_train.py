import hydra
import os
import logging
import time
from utils.common_imports import *

# Local imports
from models.hf_agent import HfAgent
from environments.dond.dond_player import DondPlayerHandler
from environments.dond.dond_game import DondGame
from models.dummy_hf_agent import DummyHfAgent
from models.oai_agent import OaiAgent
from utils.log_statistics import *
from utils.log_statistics import update_player_statistics, generate_player_stats_plots
from utils.update_start_epoch import update_start_epoch
from training.train_main import *
from generation.run_games import run_matches
import torch
import numpy as np
import random
import pickle

compute__logger = logging.getLogger("compute__logger")

def init_models(cfg, random_seed, output_directory):
    models = {}
    for model_name in cfg["models"].keys():
        if cfg["models"][model_name]["class"] == "hf":
            models[model_name] = HfAgent(**cfg["models"][model_name]["init_args"], random_seed=random_seed,
                                         output_directory=output_directory)
        elif cfg["models"][model_name]["class"] == "dummy_hf":
            models[model_name] = DummyHfAgent(**cfg["models"][model_name]["init_args"])
        elif cfg["models"][model_name]["class"] == "oai":
            models[model_name] = OaiAgent(**cfg["models"][model_name]["init_args"])
    return models


def create_blank_match(cfg, seed_offset=0):
    """
    Initializes the match for the game, ensuring that each game instance
    receives a unique random_seed passed to the DondGame, which in turn is used
    by the random generation functions for quantities and values.

    Args:
        cfg (omegaconf.DictConfig): Configuration object containing all necessary parameters.
        seed_offset (int): An offset to uniquely adjust the seed for each match.

    Returns:
        dict: A match dictionary.
    """
    players = {}
    for player_name in cfg["matches"]["players"].keys():
        players[player_name] = DondPlayerHandler(
            player_name,
            **cfg["matches"]["players"][player_name]["dond_player_args"]
        )

    # Build a fresh copy of game args to safely update random setup parameters.
    game_args = dict(cfg["matches"]["dond_game_args"])
    setup_kwargs = game_args.get("random_setup_kwargs", {})
    setup_kwargs = dict(setup_kwargs)  # shallow copy

    # Determine the base seed.
    if "seed" in setup_kwargs:
        base_seed = setup_kwargs.pop("seed")  # Remove it from kwargs so that DondGame controls randomness.
        random_seed_value = base_seed + seed_offset
    else:
        random_seed_value = random.randint(1, 10**9) + seed_offset

    # Put back the cleaned up random_setup_kwargs (without any seed key).
    game_args["random_setup_kwargs"] = setup_kwargs

    blank_match = {
        "players": players,
        "game": DondGame(
            players=list(players.keys()),
            random_seed=random_seed_value,  # Pass the unique seed here.
            **game_args
        ),
        "game_state": None,
        "stop_condition": cfg["matches"]["stop_condition"],
        "stop_condition_kwargs": cfg["matches"]["stop_condition_kwargs"]
    }
    return blank_match

def format_time(seconds):
    if seconds >= 3600:
        return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m {int(seconds % 60)}s"
    elif seconds >= 60:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    else:
        return f"{int(seconds)}s"


def dond_run_train(cfg, random_seed):
    """
    Executes a negotiation cycle for the Deal or No Deal (DoND) game.
    """
    total_start_time = time.time()

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_directory = f"{hydra_cfg['runtime']['output_dir']}/seed_{random_seed}"
    os.makedirs(output_directory, exist_ok=True)

    # Initialize models
    models = init_models(cfg, random_seed=random_seed, output_directory=output_directory)

    update_start_epoch(cfg=cfg, output_directory=output_directory)

    random.seed(random_seed)  # Python random
    np.random.seed(random_seed)  # NumPy
    torch.manual_seed(random_seed)  # PyTorch (CPU)
    torch.cuda.manual_seed(random_seed)  # PyTorch (GPU)
    torch.cuda.manual_seed_all(random_seed)  # If using multi-GPU

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
            matches.append(create_blank_match(cfg, seed_offset = (iteration * nb_matches) + i))
        players = matches[0]["players"]
        player_names = players.keys()
        run_matches(
            export_path=it_folder,
            matches=matches,
            models=models,
            **cfg['matches']['run_matches_args']
        )
        del matches

        generation_end_time = time.time()

        logging_start_time = time.time()

        for player_name in player_names:
            player_stats_folder = os.path.join(output_directory, "statistics", player_name)
            os.makedirs(player_stats_folder, exist_ok=True)
            player_stats_file = os.path.join(player_stats_folder, f"{player_name}_stats.jsonl")
            player_plots_folder = os.path.join(player_stats_folder, "plots")

            update_player_statistics(
                input_path=os.path.join(it_folder, player_name, "statistics"),
                output_file=player_stats_file
            )

            generate_player_stats_plots(
                global_stats_path=player_stats_file,
                matplotlib_log_dir=os.path.join(player_stats_folder, "matplotlib"),
                tensorboard_log_dir=os.path.join(player_stats_folder, "tensorboard"),
                wandb_log_dir=os.path.join(player_stats_folder, "wandb"),
            )

            generate_frequency_counts(os.path.join(it_folder, player_name, 'statistics'))

        logging_end_time = time.time()

        # Train models
        training_start_time = time.time()

        for model_name, model in models.items():
            if hasattr(model, 'adapters'):
                for adapter_name in model.adapters.keys():
                    mod_adpt_id = f"{model_name}/{adapter_name}"
                    model.prepare_adapter_train(adapter_name)

                    data_paths = []
                    for player in players.values():
                        if player.mod_adpt_id == mod_adpt_id:
                            player_export_path = os.path.join(it_folder, player.player_name, "training")
                            data_paths.append(player_export_path)

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

        compute__logger.info(
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
