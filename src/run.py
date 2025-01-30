import hydra
from hydra.core.hydra_config import HydraConfig
import logging
import os
import sys

from experiments.dond_run_train import dond_run_train
from experiments.arithmetic_test import arithmetic_test

@hydra.main(config_path="../conf", config_name="default")
def main(cfg):


    # Get Hydra's runtime directory
    hydra_run_dir = HydraConfig.get().run.dir

    # Define specific loggers to configure
    specific_loggers =[
        "model_logger",
        "compute__logger",
        "memory_logger",
        "games_logger",
    ]

    # Dynamically configure handlers for specific loggers
    for logger_name in specific_loggers:
        logger = logging.getLogger(logger_name)
        log_file = os.path.join(hydra_run_dir, f"{logger_name}.log")
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"))
        logger.addHandler(handler)
        logger.propagate = False  # Prevent duplicate logs

    # Redirect stdout and stderr to root logger
    root_logger = logging.getLogger("root")
    sys.stdout = LoggerStream(root_logger.info)
    sys.stderr = LoggerStream(root_logger.error)

    # Set the variable 'game_intro_prompt' according to 'max_turns'
    if cfg["matches"]["dond_game_args"]["max_turns"] == 1:
        cfg["prompt"]["game_intro_prompt"] = cfg["prompt"]["1_turn_prompt"]
    else:
        cfg["prompt"]["game_intro_prompt"] = cfg["prompt"]["multi_turn_prompt"]

    # Run the experiment specified in the configuration
    globals()[cfg.experiment.method](cfg)


class LoggerStream:
    """
    Helper class to redirect stdout and stderr to a logger.
    """
    def __init__(self, log_func):
        self.log_func = log_func

    def write(self, message):
        if message.strip():
            self.log_func(message.strip())

    def flush(self):
        pass


if __name__ == "__main__":
    main()
