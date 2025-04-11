import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from src.utils.log_statistics import generate_agent_stats_plots

# path at which the statistics folder contains, {player}_stats.jsonl files
output_directory = (
    "/home/mila/u/udaykaran.kapur/scratch/outputs/2025-04-09/21-19-31_seed_33/seed_33"
)

for agent_name in ["alice", "bob"]:
    agent_stats_folder = os.path.join(output_directory, "statistics", agent_name)

    agent_stats_file = os.path.join(agent_stats_folder, f"{agent_name}_stats.jsonl")

    generate_agent_stats_plots(
        global_stats_path=agent_stats_file,
        matplotlib_log_dir=os.path.join(agent_stats_folder, "matplotlib"),
        tensorboard_log_dir=os.path.join(agent_stats_folder, "tensorboard"),
        wandb_log_dir=os.path.join(agent_stats_folder, "wandb"),
    )
