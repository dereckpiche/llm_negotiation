import argparse
import os
import sys

# Sample usage:
# python scripts/generate_adhoc_plots.py --output_directory /home/mila/u/udaykaran.kapur/scratch/outputs/2025-04-09/21-19-31_seed_33/seed_33

# Add project root to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from mllm.utils.log_statistics import generate_agent_stats_plots

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Generate agent statistics plots.")
parser.add_argument(
    "--output_directory",
    type=str,
    required=True,
    help="Path to the output directory containing the 'statistics' folder.",
)

args = parser.parse_args()
output_directory = args.output_directory

for agent_name in ["alice", "bob"]:
    agent_stats_folder = os.path.join(output_directory, "statistics", agent_name)

    agent_stats_file = os.path.join(agent_stats_folder, f"{agent_name}_stats.jsonl")

    generate_agent_stats_plots(
        global_stats_path=agent_stats_file,
        matplotlib_log_dir=os.path.join(agent_stats_folder, "matplotlib"),
        tensorboard_log_dir=os.path.join(agent_stats_folder, "tensorboard"),
        wandb_log_dir=os.path.join(agent_stats_folder, "wandb"),
    )
