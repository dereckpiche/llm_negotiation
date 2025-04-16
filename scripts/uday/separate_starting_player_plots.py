import argparse
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from src.utils.log_statistics import generate_agent_stats_plots

for agent_name in ["alice", "bob"]:
    for i in ["first", "second"]:
        agent_stats_folder = os.path.join(
            "/home/mila/u/udaykaran.kapur/llm_negotiation/seed_500_temp",
            f"{agent_name}_{i}",
        )

        agent_stats_file = os.path.join(
            agent_stats_folder, f"{agent_name}_{i}_stats.json"
        )

        generate_agent_stats_plots(
            global_stats_path=agent_stats_file,
            matplotlib_log_dir=os.path.join(agent_stats_folder, "matplotlib"),
            tensorboard_log_dir=os.path.join(agent_stats_folder, "tensorboard"),
            wandb_log_dir=os.path.join(agent_stats_folder, "wandb"),
        )
