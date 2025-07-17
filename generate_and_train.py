"""
This file contains the code to generate and train the models.
"""
import copy, logging, os, shutil, sys, hydra
from hydra.core.hydra_config import HydraConfig
from typing import Dict, List, Any
import asyncio
from pandas.core.base import IndexLabel
from mllm.models.dummy_local_llm import DummyLocalLLM
from mllm.models.local_llm import LocalLLM
from mllm.models.lean_local_llm import LeanLocalLLM
from mllm.models.server_llm import ServerLLM
from mllm.models.critic_wrapper import ScalarCritic
from mllm.training.reinforce_trainer import *
from mllm.training.reinforce_trainer_tally import *
from mllm.utils.common_imports import *
from mllm.utils.update_start_epoch import update_start_epoch
from mllm.utils.get_stochastic_game_lengths import get_stochastic_game_lengths
from mllm.utils.dict_get_path import get_from_nested_dict
from mllm.markov_games.mg_utils import AgentConfig, MarkovGameConfig, init_markov_game_components
from mllm.markov_games.runners.alternative_actions_runner import AlternativeActionsRunner
from mllm.markov_games.runners.linear_runner import LinearRunner
from mllm.markov_games.run_markov_games import run_markov_games


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

    # Init llms + llm adapters
    llms_dict = {}
    for llm_id, model_config in cfg["models"].items():
        model_class = globals()[model_config["class"]]
        llms_dict[llm_id] = model_class(
            **model_config["init_args"],
            output_directory=output_directory,
        )

    # Get dictionnary of functionnal-like callable policies (only for inference)
    policies = {}
    for llm_id, llm in llms_dict.items():
        policies.update(llm.get_callable_objects())

    # Get dictionnary of pytorch-like trainable models
    trainable_modules = {}
    for llm_id, llm in llms_dict.items():
        trainable_modules[llm_id] = llm.get_trainable_objects()

    # Critics
    for critic_id, critic_config in cfg["critics"].items():
        pointer = critic_config["pointer"]
        base = get_from_nested_dict(trainable_modules, pointer)
        trainable_modules[critic_id] = ScalarCritic(base)

    # Init optimizers
    optimizers = {}
    for optimizer_id, optimizer_config in cfg["optimizers"].items():
        pointer = optimizer_config["pointer"]
        base = get_from_nested_dict(trainable_modules, pointer)
        optimizer_class = eval(optimizer_config["optimizer_class"])
        init_args = optimizer_config["init_args"]
        optimizers[optimizer_id] = optimizer_class(base.parameters(), **init_args)

    # Init trainers
    trainers = {}
    for trainer_id, trainer_config in cfg["trainers"].items():
        pointers = trainer_config["pointers"]
        tokenizer = llms_dict[pointers["model"][0]].tokenizer
        policy_model = get_from_nested_dict(trainable_modules, pointers["model"])
        policy_optimizer = get_from_nested_dict(optimizers, pointers["optimizer"])
        if pointers.get("critic", True):
            critic_model = get_from_nested_dict(
                trainable_modules, pointers["critic"])
        else: critic_model = None
        if pointers.get("critic_optimizer", True):
            critic_optimizer = get_from_nested_dict(
                optimizers, pointers["critic_optimizer"])
        else: critic_optimizer = None
        trainer = ReinforceTrainerWRS(
                    model=policy_model,
                    optimizer=policy_optimizer,
                    tokenizer=tokenizer,
                    lr_scheduler=None, # TODO add
                    critic=critic_model,
                    critic_optimizer=critic_optimizer,
                    critic_lr_scheduler=None, # TODO add
                    config=RtConfig(
                        **trainer_config["init_args"],
                        logging_path=None,
                    ),
                    save_path=os.path.join(output_directory, trainer_id)
                )
        trainers[trainer_id] = trainer



    for iteration in range(cfg["experiment"]["start_epoch"], cfg["experiment"]["nb_epochs"]):

        # -----------------------------------------------------------------
        # Create and run Markov Games
        # -----------------------------------------------------------------
        for llm in llms_dict.values():
            llm.toggle_eval_mode()

        # Create a new RNG instance by splitting the current one (simulates RNG splitting)
        env_rng = np.random.default_rng(env_rng.integers(0, 1e9))
        iteration_start_time = time.time()
        it_folder = os.path.join(output_directory, f"iteration_{iteration:03}")
        os.makedirs(it_folder, exist_ok=True)
        generation_start_time = time.time()

        # TODO: maybe only create these once and then use reset!
        # Create markov games
        agent_configs = []
        for agent_config_ in cfg["markov_games"]["agents"].values():
            agent_config = AgentConfig(**agent_config_)
            agent_configs.append(agent_config)
        markov_game_config = MarkovGameConfig(
            id = "",
            seed = 0,
            simulation_class_name = cfg["markov_games"]["simulation_class_name"],
            simulation_init_args = cfg["markov_games"]["simulation_init_args"],
            agent_configs = agent_configs,
            output_path = ""
        )
        markov_games = []
        nb_matches = cfg["experiment"]["nb_matches_per_iteration"]
        for i in range(nb_matches):
            markov_game_config.seed = int(env_rng.integers(0, 1e9))
            markov_game_id = "mgid_" + str(iteration*nb_matches+i)
            markov_game_config.id = markov_game_id
            markov_game_config.output_path = os.path.join(it_folder, str(iteration*nb_matches+i))
            markov_game = init_markov_game_components(config=markov_game_config, policies=policies)
            markov_games.append(markov_game)

        # Generate rollouts raw data (using asyncio)
        runner = eval(cfg["markov_games"]["runner_method_name"])
        # TODO: throw error if error in asyncio call
        await run_markov_games(runner=runner, markov_games=markov_games)

        print("PHASE 1 DONE!")
        generation_end_time = time.time()

        # Process raw data into training data using the specified functions for each agent


        # -----------------------------------------------------------------
        # Train
        # -----------------------------------------------------------------


        # Generate training data files from raw data
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
        for llm in llms_dict.values():
            llm.toggle_training_mode()


        # Training Phase 1: set training data, opp shaping info transfers
        shaping_info_sets = []
        for trainer_id, trainer in trainers.items():
            trainer.set_training_data(
                paths = trainer_training_paths[trainer_id]
            )
            info = trainer.send_trainer_info()
            shaping_info_sets.append(info)

        # Training Phase 2: use opp shaping infos, apply reinforce
        trainer_items = list(trainers.items())
        for i, (trainer_id, trainer) in enumerate(trainer_items):
            info = shaping_info_sets[len(shaping_info_sets)-1-i]
            trainer.use_co_trainer_info(
                co_trainer_info=info
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
        for llm in llms_dict.values(): llm.export_adapters()

        # Export optimizer states
        for trainer in trainers.values():
            trainer.export_optimizer_states()


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

    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    compute_logger.info(
        f"Total time taken for the entire run: {format_time(total_duration)}"
    )




@hydra.main()
def main(cfg):
    # Get Hydra's runtime directory
    hydra_run_dir = HydraConfig.get().run.dir

    # Output source code in runtime directory for certain reproducibility
    os.makedirs(hydra_run_dir, exist_ok=True)
    shutil.copytree(
        "mllm",
        os.path.join(hydra_run_dir, "src_code_for_reproducibility"),
        dirs_exist_ok=True,
    )

    # Run the experiment specified in the configuration
    asyncio.run(generate_and_train(
        OmegaConf.to_container(cfg, resolve=True, structured_config_mode="dict"),
        base_seed=cfg.experiment.base_seed,
    ))

    print(f"Run for seed_{cfg.experiment.base_seed} complete!")




if __name__ == "__main__":
    main()
