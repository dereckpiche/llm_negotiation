import wandb
import os

def upload_tensorboard_logs(tensorboard_log_dir, project_name, run_name):
    """
    Upload TensorBoard logs to W&B and visualize the plots in the W&B run interface.

    Args:
        tensorboard_log_dir (str): Path to the TensorBoard log directory.
        project_name (str): Name of the W&B project.
        run_name (str): Name of the W&B run.
    """
    # Initialize a W&B run
    wandb.init(project=project_name, name=run_name)

    # Ensure TensorBoard integration
    wandb.tensorboard.patch(root_logdir=tensorboard_log_dir)

if __name__ == "__main__":
    # Example usage
    tensorboard_log_dir = "/home/mila/d/dereck.piche/llm_negotiation/outputs/2025-01-16/14-48-38/statistics/alice/tensorboard"  # Path to your TensorBoard logs folder
    project_name = "my_project"      # Your W&B project name
    run_name = "my_tensorboard_run222" # Your desired run name

    upload_tensorboard_logs(tensorboard_log_dir, project_name, run_name)