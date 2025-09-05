"""
This file contains the code to generate and train the models.
TODO: don't use any eval() (maybe switch to gin configs instead of hydra)
TODO: use ModulePointer instead of nested dicts
"""
import asyncio
import copy
import logging
import os
import pickle
import random
import re
import shutil
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from pandas.core.base import IndexLabel

from mllm.markov_games.alternative_actions_runner import AlternativeActionsRunner
from mllm.markov_games.group_timesteps import group_by_round
from mllm.markov_games.linear_runner import LinearRunner
from mllm.markov_games.mg_utils import (
    AgentConfig,
    MarkovGameConfig,
    init_markov_game_components,
)
from mllm.markov_games.run_markov_games import run_markov_games
from mllm.models.large_language_model_api import LargeLanguageModelOpenAI
from mllm.models.large_language_model_local import LeanLocalLLM

# from mllm.models.large_language_model_server import ServerLLM
from mllm.models.scalar_critic import ScalarCritic
from mllm.training.trainer_ad_align import TrainerAdAlign
from mllm.training.trainer_independent import TrainerNaive
from mllm.training.trainer_sum_rewards import TrainerSumRewards
from mllm.utils.dict_get_path import get_from_nested_dict
from mllm.utils.kill_sglang import kill_sglang
from mllm.utils.resource_context import resource_logger_context
from mllm.utils.short_id_gen import generate_short_id
from mllm.utils.update_start_epoch import update_start_epoch

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))


@dataclass
class ModulePointer:
    base_llm_id: str
    adapter_id: str


async def generate_and_train(cfg: dict, base_seed: int) -> None:
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

    # Init llms + llm adapters
    llms_dict = {}
    for llm_id, model_config in cfg["models"].items():
        model_class: LeanLocalLLM | LargeLanguageModelOpenAI = globals()[
            model_config["class"]
        ]  # TODO: Add server llm
        llms_dict[llm_id] = model_class(
            **model_config["init_args"],
            output_directory=output_directory,
        )

    adapter_modules = {}  # These are trainable Pytorch modules
    for llm_id, llm in llms_dict.items():
        if isinstance(llm, LeanLocalLLM):
            adapter_modules[llm_id] = llm.get_adapter_modules()

    # Scalar Critics
    critics = {}
    for critic_id, critic_config in cfg["critics"].items():
        critic_module_pointer = critic_config["module_pointer"]
        critic_adapter = get_from_nested_dict(adapter_modules, critic_module_pointer)
        critics[critic_id] = ScalarCritic(critic_adapter)

    trainable_modules = {**adapter_modules, **critics}

    # Init optimizers
    optimizers = {}
    for optimizer_id, optimizer_config in cfg["optimizers"].items():
        optimizer_module_pointer = optimizer_config["module_pointer"]
        module = get_from_nested_dict(trainable_modules, optimizer_module_pointer)
        optimizer_class: torch.optim.Adam | torch.optim.SGD = eval(
            optimizer_config["optimizer_class_name"]
        )
        init_args = optimizer_config["init_args"]
        optimizers[optimizer_id] = optimizer_class(module.parameters(), **init_args)

    # Init trainers
    trainers = {}
    for trainer_id, trainer_config in cfg["trainers"].items():
        trainer_class = eval(trainer_config["class"])
        module_pointers = trainer_config["module_pointers"]
        tokenizer = llms_dict[module_pointers["policy"][0]].tokenizer
        policy = get_from_nested_dict(adapter_modules, module_pointers["policy"])
        policy_optimizer = get_from_nested_dict(
            optimizers, module_pointers["policy_optimizer"]
        )
        if module_pointers.get("critic", False):
            critic = get_from_nested_dict(critics, module_pointers["critic"])
        else:
            critic = None
        if module_pointers.get("critic_optimizer", False):
            critic_optimizer = get_from_nested_dict(
                optimizers, module_pointers["critic_optimizer"]
            )
        else:
            critic_optimizer = None
        trainer: TrainerAdAlign | TrainerNaive | TrainerSumRewards = trainer_class(
            policy=policy,
            policy_optimizer=policy_optimizer,
            critic=critic,
            critic_optimizer=critic_optimizer,
            tokenizer=tokenizer,
            lr_scheduler=None,  # TODO add
            critic_lr_scheduler=None,  # TODO add
            save_path=os.path.join(output_directory, trainer_id),
            **trainer_config["kwargs"],
        )
        trainers[trainer_id] = trainer

        # Stuff common across iterations
        agent_configs = []
        for agent_config_ in cfg["markov_games"]["agents"].values():
            agent_config = AgentConfig(**agent_config_)
            agent_configs.append(agent_config)

        nb_matches = cfg["experiment"]["nb_matches_per_iteration"]
        seed_group_size = cfg["experiment"].get("seed_group_size", 1)
        assert (
            nb_matches % seed_group_size == 0
        ), "nb_matches must be divisible by seed_group_size"

    for iteration in range(
        cfg["experiment"]["start_epoch"], cfg["experiment"]["nb_epochs"]
    ):
        logger.info(f"Starting iteration {iteration}.")
        # -----------------------------------------------------------------
        # Create and run Markov Games
        # -----------------------------------------------------------------
        for llm in llms_dict.values():
            await llm.toggle_eval_mode()
        # Get dictionnary of functionnal-like callable policies (only for inference)
        policies = {}
        for llm_id, llm in llms_dict.items():
            policies.update(llm.get_inference_policies())

        # Set folders and seeds
        env_rng = np.random.default_rng(env_rng.integers(0, 1e9))
        crn_rng = copy.deepcopy(env_rng)
        iteration_start_time = time.time()
        it_folder = os.path.join(output_directory, f"iteration_{iteration:03}")
        crn_seeds = [
            crn_rng.integers(0, 1e9, 1)[0] for _ in range(nb_matches // seed_group_size)
        ]  # common random number seeds
        os.makedirs(it_folder, exist_ok=True)
        generation_start_time = time.time()

        # Create new markov games
        markov_games = []
        for match_number in range(nb_matches):
            env_rng = np.random.default_rng(env_rng.integers(0, 1e9))
            match_rng = copy.deepcopy(env_rng)

            def agent_configs_per_match(agent_configs, match_number, match_rng):
                new_agent_configs = []
                for index, agent_config in enumerate(agent_configs):
                    if (match_number % len(agent_configs)) == index:
                        new_agent_configs.append(agent_config)
                    else:
                        # take_buffer_agent = match_rng.choice([True, False])
                        take_buffer_agent = True
                        if take_buffer_agent:
                            buffer_agent_config = copy.deepcopy(agent_config)
                            buffer_agent_config.agent_id = (
                                f"buffer_{agent_config.agent_id}"
                            )
                            policy_ids = list(policies.keys())
                            buffer_policy_ids = [
                                policy_id
                                for policy_id in policy_ids
                                if "buffer" in policy_id
                                and agent_config.policy_id in policy_id
                            ]
                            buffer_policy_ids = sorted(
                                buffer_policy_ids,
                                key=lambda x: int(re.search(r"iter_(\d+)", x).group(1)),
                                reverse=True,
                            )[: cfg["experiment"]["agent_buffer_recent_k"]]
                            buffer_agent_config.policy_id = match_rng.choice(
                                buffer_policy_ids
                            )
                            if buffer_agent_config.policy_id is None:
                                new_agent_configs.append(agent_config)
                            else:
                                new_agent_configs.append(buffer_agent_config)
                        else:
                            new_agent_configs.append(agent_config)
                return new_agent_configs

            markov_game_config = MarkovGameConfig(
                id=iteration * nb_matches + match_number,
                seed=int(crn_seeds[match_number // seed_group_size]),
                simulation_class_name=cfg["markov_games"]["simulation_class_name"],
                simulation_init_args=cfg["markov_games"]["simulation_init_args"],
                agent_configs=agent_configs_per_match(
                    agent_configs, match_number, match_rng
                )
                if cfg["experiment"].get("agent_buffer", False)
                else agent_configs,
            )
            markov_game = init_markov_game_components(
                config=markov_game_config, policies=policies
            )
            markov_games.append(markov_game)

        # Generate rollouts raw data asynchronously
        runner = eval(cfg["markov_games"]["runner_method_name"])
        rollout_trees = await run_markov_games(
            runner=runner,
            runner_kwargs=cfg["markov_games"]["runner_kwargs"],
            output_folder=it_folder,
            markov_games=markov_games,
        )
        # This will merge all timesteps of a round into a single timestep - simplifies credit assignment during training
        if cfg["markov_games"].get("group_by_round", False):
            rollout_trees = [
                group_by_round(rollout_tree) for rollout_tree in rollout_trees
            ]

        # Export rollout trees
        for i, rollout_tree in enumerate(rollout_trees):
            with open(
                os.path.join(it_folder, f"mgid_{rollout_tree.id}.rt.pkl"), "wb"
            ) as f:
                # Store as pure Python dict to avoid class dependency on load
                pickle.dump(
                    rollout_tree.model_dump(), f, protocol=pickle.HIGHEST_PROTOCOL
                )

        generation_end_time = time.time()

        # Process raw data into training data using the specified functions for each agent

        # -----------------------------------------------------------------
        # Train
        # -----------------------------------------------------------------
        if not cfg["experiment"]["train"]:
            continue

        training_start_time = time.time()

        # Prepare base models for training
        for llm in llms_dict.values():
            await llm.toggle_training_mode()

        # ----------- Training (with advantage sharing between trainers)
        # Send advantage packets to other trainers
        all_advantage_packets = []
        for trainer_id, trainer in trainers.items():
            trainer.set_trajectory_data(
                rollout_trees=rollout_trees,
                agent_ids=cfg["train_on_which_data"][trainer_id],
            )
            advantage_packets = trainer.share_advantage_data()
            all_advantage_packets.extend(advantage_packets)

        # Receive advantage packets from other trainers and train
        for trainer_id, trainer in trainers.items():
            trainer.receive_advantage_data(all_advantage_packets)
            trainer.set_policy_gradient_data()
            trainer.train()

        # Export trainer stuff
        for trainer_id, trainer in trainers.items():
            trainer.export_training_tally(
                identifier=trainer_id,
                folder=it_folder,
            )
            trainer.export_trainer_states()

        # Export all HF adapters weights (needed for vLLM inference)
        for llm in llms_dict.values():
            llm.export_adapters()

        training_end_time = time.time()

        # Checkpoint all adapters
        checkpoint_frequency = cfg["experiment"]["checkpoint_every_n_iterations"]
        if (
            checkpoint_frequency != -1
            and iteration % checkpoint_frequency == 0
            and iteration != 0
        ):
            for llm_id, llm in llms_dict.items():
                if hasattr(llm, "adapter_paths"):
                    llm.checkpoint_all_adapters(
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

        def format_time(seconds):
            if seconds >= 3600:
                return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m {int(seconds % 60)}s"
            elif seconds >= 60:
                return f"{int(seconds // 60)}m {int(seconds % 60)}s"
            else:
                return f"{int(seconds)}s"

        logger.info(
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


@hydra.main(config_path="./configs")
def main(cfg):
    # Get Hydra's runtime directory
    hydra_run_dir = HydraConfig.get().run.dir
    filename = os.path.join(hydra_run_dir, "generate_and_train_log.log")
    logging.basicConfig(filename=filename, level=logging.INFO)

    # Output source code in runtime directory for certain reproducibility
    os.makedirs(hydra_run_dir, exist_ok=True)
    shutil.copytree(
        "mllm",
        os.path.join(hydra_run_dir, "src_code_for_reproducibility"),
        dirs_exist_ok=True,
    )

    # Run the experiment specified in the configuration
    try:
        asyncio.run(
            generate_and_train(
                OmegaConf.to_container(
                    cfg, resolve=True, structured_config_mode="dict"
                ),
                base_seed=cfg.experiment.base_seed,
            )
        )
    finally:
        # Clean up distributed process groups if they exist
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    kill_sglang()
    main()
