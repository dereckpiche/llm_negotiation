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

from mllm.markov_games.mg_utils import (
    AgentConfig,
    MarkovGameConfig,
    init_markov_game_components,
)
from mllm.markov_games.run_markov_games import run_markov_games
from mllm.markov_games.runners.alternative_actions_runner import (
    AlternativeActionsRunner,
)
from mllm.markov_games.runners.linear_runner import LinearRunner
from mllm.models.large_language_model_local import LeanLocalLLM

# from mllm.models.large_language_model_server import ServerLLM
from mllm.models.scalar_critic import ScalarCritic
from mllm.training.advantage_alignment_trainer import AdAlignTrainer
from mllm.training.reinforce_trainer import BaseTrainer
from mllm.utils.dict_get_path import get_from_nested_dict
from mllm.utils.kill_sglang import kill_sglang
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
        model_class: LeanLocalLLM = globals()[  # TODO: Add server llm
            model_config["class"]
        ]
        llms_dict[llm_id] = model_class(
            **model_config["init_args"],
            output_directory=output_directory,
        )

    # Get dictionnary of functionnal-like callable policies (only for inference)
    policies = {}
    for llm_id, llm in llms_dict.items():
        policies.update(llm.get_inference_policies())

    # -----------------------------------------------------------------
    # Create and run Markov Games
    # -----------------------------------------------------------------

    for llm in llms_dict.values():
        llm.toggle_eval_mode()

    # Create a new RNG instance by splitting the current one (simulates RNG splitting)
    env_rng = np.random.default_rng(env_rng.integers(0, 1e9))

    # TODO: maybe only create these once and then use reset!
    # Create markov games
    agent_configs = []
    for agent_config_ in cfg["markov_games"]["agents"].values():
        agent_config = AgentConfig(**agent_config_)
        agent_configs.append(agent_config)
    markov_game_config = MarkovGameConfig(
        id="",
        seed=0,
        simulation_class_name=cfg["markov_games"]["simulation_class_name"],
        simulation_init_args=cfg["markov_games"]["simulation_init_args"],
        agent_configs=agent_configs,
    )
    markov_games = []
    nb_matches = cfg["experiment"]["nb_matches_per_iteration"]
    for i in range(nb_matches):
        markov_game_config.seed = int(env_rng.integers(0, 1e9))
        markov_game = init_markov_game_components(
            config=markov_game_config, policies=policies
        )
        markov_games.append(markov_game)

    # Generate rollouts raw data (using asyncio)
    runner = eval(cfg["markov_games"]["runner_method_name"])
    # TODO: throw error if error in asyncio call
    rollout_trees = await run_markov_games(
        runner=runner,
        runner_kwargs=cfg["markov_games"]["runner_kwargs"],
        output_folder=output_directory,
        markov_games=markov_games,
    )


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
    asyncio.run(
        generate_and_train(
            OmegaConf.to_container(cfg, resolve=True, structured_config_mode="dict"),
            base_seed=cfg.experiment.base_seed,
        )
    )


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    kill_sglang()
    asyncio.run(main())
