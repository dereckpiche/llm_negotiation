import hydra
from hydra.core.hydra_config import HydraConfig
import logging
import os
import sys
from omegaconf import OmegaConf
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

    print("Total epochs: ", cfg.experiment.nb_epochs)

    # Dynamically configure handlers for specific loggers
    for logger_name in specific_loggers:
        logger = logging.getLogger(logger_name)
        log_dir = os.path.join(hydra_run_dir, f"seed_{cfg.experiment.base_seed}")  # Extract directory path
        os.makedirs(log_dir, exist_ok=True)  # Ensure directory exists

        log_file = os.path.join(log_dir, f"{logger_name}.log")
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"))
        logger.addHandler(handler)
        logger.propagate = False  # Prevent duplicate logs

    # Redirect stdout and stderr to root logger
    root_logger = logging.getLogger(f"root")
    sys.stdout = LoggerStream(root_logger.info)
    sys.stderr = LoggerStream(root_logger.error)

    # Run the experiment specified in the configuration
    globals()[cfg.experiment.method](OmegaConf.to_container(cfg, resolve=True, structured_config_mode="dict"), base_seed=cfg.experiment.base_seed)

    print(f"Run for seed_{cfg.experiment.base_seed} complete!")


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
