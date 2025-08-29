"""

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

from mllm.markov_games.linear_runner import LinearRunner
from mllm.markov_games.mg_utils import (
    AgentConfig,
    MarkovGameConfig,
    init_markov_game_components,
)
from mllm.markov_games.run_markov_games import run_markov_games
from mllm.models.large_language_model_local import LeanLocalLLM

# from mllm.models.large_language_model_server import ServerLLM
from mllm.utils.kill_sglang import kill_sglang

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))


@dataclass
class ModulePointer:
    base_llm_id: str
    adapter_id: str


async def faceoff(cfg: dict, base_seed: int) -> None:
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

    random.seed(base_seed)  # Python random
    np.random.seed(base_seed)  # NumPy
    torch.manual_seed(base_seed)  # PyTorch (CPU)
    torch.cuda.manual_seed(base_seed)  # PyTorch (GPU)
    torch.cuda.manual_seed_all(base_seed)  # If using multi-GPU

    env_rng = np.random.default_rng(base_seed)

    # -----------------------------------------------------------------
    # Initialize models
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
    for _ in range(nb_matches):
        markov_game_config.seed = int(env_rng.integers(0, 1e9))
        markov_game = init_markov_game_components(
            config=markov_game_config, policies=policies
        )
        markov_games.append(markov_game)

    # Faceoff models
    runner = eval(cfg["markov_games"]["runner_method_name"])
    rollout_trees = await run_markov_games(
        runner=runner,
        runner_kwargs=cfg["markov_games"]["runner_kwargs"],
        output_folder=output_directory,
        markov_games=markov_games,
    )
    # Export rollout trees
    for i, rollout_tree in enumerate(rollout_trees):
        with open(os.path.join(it_folder, f"mgid_{i}_rollout_tree.json"), "w") as f:
            f.write(rollout_tree.model_dump_json(indent=4))


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

    try:
        # Run the experiment specified in the configuration
        asyncio.run(
            faceoff(
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
