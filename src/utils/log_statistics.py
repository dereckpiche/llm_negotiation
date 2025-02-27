
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict
# import wandb
# from comet_ml import Experiment

# experiment = Experiment(
#     api_key="IvI06nn59lLap4y0JRrwlTViy",
#     project_name="llm_negotiation",
#     log_env_gpu=True,
#     log_env_cpu=True
# )

def append_statree(tree1: Dict, tree2: Dict):
    """
    Append each corresponding leaf of tree2 to tree1.
    """
    for key, value in tree2.items():
        if key not in tree1:
            tree1[key] = value
        elif isinstance(value, dict):
            append_statree(tree1[key], value)
        elif isinstance(value, list):
            if isinstance(tree1[key], list):
                tree1[key].extend(value)
            else:
                tree1[key] = [tree1[key]] + value
        else:
            if isinstance(tree1[key], list):
                tree1[key].append(value)
            else:
                tree1[key] = [tree1[key], value]

def get_mean_statree(tree: Dict) -> Dict:
    """
    scores a statree where each leaf is replaced by the mean of the leaf values.
    """
    result = {}
    for key, value in tree.items():
        if isinstance(value, dict):
            result[key] = get_mean_statree(value)
        elif isinstance(value, list):
            cleaned_values = [v for v in value if v is not None]
            result[key] = np.mean(cleaned_values) if cleaned_values else None
        else:
            result[key] = value
    return result

def get_var_statree(tree: Dict) -> Dict:
    """
    scores a statree where each leaf is replaced by the variance of the leaf values.
    """
    result = {}
    for key, value in tree.items():
        if isinstance(value, dict):
            result[key] = get_var_statree(value)
        elif isinstance(value, list):
            cleaned_values = [v for v in value if v is not None]
            result[key] = np.var(cleaned_values) if cleaned_values else None
        else:
            result[key] = 0  # Single value has variance 0
    return result

def plot_statree(tree: Dict, folder: str, path: str = ""):
    """
    Plots the leaves of the statree and saves them to the specified folder.
    """
    os.makedirs(folder, exist_ok=True)
    for key, value in tree.items():
        new_path = f"{path}/{key}" if path else key
        if isinstance(value, dict):
            plot_statree(value, folder, new_path)
        elif isinstance(value, list):
            plt.figure()
            plt.plot(value)
            plt.title(new_path)
            plt.savefig(os.path.join(folder, f"{new_path.replace('/', '_')}.png"))
            plt.close()

def tb_statree(tree: Dict, writer, path: str = ""):
    """
    Logs the leaves of the statree to TensorBoard.
    """
    for key, value in tree.items():
        new_path = f"{path}/{key}" if path else key
        if isinstance(value, dict):
            tb_statree(value, writer, new_path)
        elif isinstance(value, list):
            for i, v in enumerate(value):
                if v is not None:
                    writer.add_scalar(new_path, v, i)



def update_player_statistics(input_path, output_file):
    """
    Computes statistics for the current iteration and updates the global statistics file.

    Args:
        input_path (str): Path to the folder containing player JSON files for the current iteration.
        output_file (str): Path to the JSON file where statistics are stored.
    """

    # Build statree by appending each dict from JSON files in "input_path" folder
    statree = {}
    for filename in os.listdir(input_path):
        if filename.endswith('.json'):
            with open(os.path.join(input_path, filename), 'r') as f:
                data = json.load(f)
                append_statree(statree, data)
    # Get epoch mean statree
    mean_statree = get_mean_statree(statree)

    # Add mean statree to global stats file
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            global_stats = json.load(f)
    else:
        global_stats = {}

    append_statree(global_stats, mean_statree)

    with open(output_file, 'w') as f:
        json.dump(global_stats, f, indent=4)



def generate_player_stats_plots(global_stats_path, matplotlib_log_dir, tensorboard_log_dir, wandb_log_dir):
    """
    Visualizes the global statistics by logging them to TensorBoard and Weights & Biases.

    Args:
        global_stats_path (str): Path to the global statistics JSON file.
        tensorboard_log_dir (str): Directory to save TensorBoard logs.
        wandb_log_dir (str): Directory for Weights & Biases run metadata.
    """
    os.makedirs(matplotlib_log_dir, exist_ok=True)
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    os.makedirs(wandb_log_dir, exist_ok=True)



    with open(global_stats_path, 'r') as f:
        global_stats = json.load(f)

    plot_statree(global_stats, folder=matplotlib_log_dir)

    # Log statistics to TensorBoard
    writer = SummaryWriter(tensorboard_log_dir)
    tb_statree(global_stats, writer)
    writer.close()

    #wb_statree(global_stats)

def generate_frequency_counts(input_path):
    agreement_percent_values = []
    items_given_to_self_values = []

    for filename in os.listdir(input_path):
        if filename.endswith('.json'):
            file_path = os.path.join(input_path, filename)
            with open(file_path, 'r') as f:
                data = json.load(f)

            for rounds, values in data.items():

                agreement_percent_values.append(values['agreement_percentage'])
                items_given_to_self_values.append(values['items_given_to_self'])

    # Convert lists to frequency counts
    agreement_percent_freq_counts = dict(Counter(agreement_percent_values))
    items_given_to_self_freq_counts = dict(Counter(items_given_to_self_values))

    # Combine into a final dictionary
    freq_stats = {
        "agreement_percent_freq": agreement_percent_freq_counts,
        "items_given_to_self_freq": items_given_to_self_freq_counts
    }

    output_path = os.path.join(input_path, 'frequency_stats.json')
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(freq_stats, f, indent=4)

def plot_seed_averaged_stats(root_path, player_names):
    """
    Plots seed-averaged statistics for given players.

    Args:
        root_path (str): Path to the directory containing seed data.
        player_names (list): List of player names to process.
    """

    # Initialize data structure for storing statistics
    player_stats = {player: {} for player in player_names}

    # Identify all seed directories
    seed_dirs = [dir_name for dir_name in os.listdir(root_path) if dir_name.startswith("seed")]

    for player in player_names:
        # Create output directory for averaged stats
        avg_stats_dir = os.path.join(root_path, "avg_seed_stats", player)
        os.makedirs(avg_stats_dir, exist_ok=True)

        # Collect statistics from each seed directory
        for seed_dir in seed_dirs:
            stats_file = os.path.join(root_path, seed_dir, "statistics", player, f"{player}_stats.jsonl")

            with open(stats_file, "r") as file:
                json_data = json.load(file)

            for round_id, round_data in json_data.items():
                for metric, values in round_data.items():
                    # Initialize nested dictionary structure
                    player_stats.setdefault(player, {}).setdefault(round_id, {}).setdefault(metric, []).append({
                        'data': [val if val is not None else 0 for val in values],
                        'file': seed_dir
                    })

        # Save collected statistics to a JSON file
        json_output_file = os.path.join(avg_stats_dir, f"{player}_aggregated_stats.json")
        with open(json_output_file, "w") as json_file:
            json.dump(player_stats[player], json_file, indent=4)

        # Plot and save seed-averaged statistics
        for round_id, round_metrics in player_stats[player].items():
            for metric, metric_data in round_metrics.items():
                plt.figure()

                # Convert data into a NumPy array for processing
                metric_data = np.array([entry['data'] for entry in metric_data])
                metric_mean = np.mean(metric_data, axis=0)
                metric_std = np.std(metric_data, axis=0)

                # Plot individual runs with pale blue
                for instance in metric_data:
                    plt.plot(instance, color="lightblue", alpha=0.5)

                # Overlay mean curve with dark blue
                plt.plot(metric_mean, linewidth=2, color="darkblue", label=f"Average")

                # Formatting and saving the plot
                plt.title(f"{round_id}/seed_averaged_{metric}")
                plt.xlabel("Iterations")
                plt.ylabel(metric.replace("_", " ").title())
                plt.legend()

                output_filename = os.path.join(avg_stats_dir, f"{round_id}_seed_averaged_{metric}.png")
                plt.savefig(output_filename)
                plt.close()  # Close figure to free memory

                # Compute standard error
                plt.figure()
                std_error = metric_std / np.sqrt(len(metric_mean))
                plt.plot(metric_mean, linestyle="-", linewidth=2, color="darkblue")

                plt.errorbar(range(len(metric_mean)), metric_mean, yerr=std_error, fmt="o", color="#006400", capsize=3, markersize=3)

                plt.title(f"{round_id}/std_error_{metric}")
                plt.xlabel("Iterations")
                plt.ylabel(metric.replace("_", " ").title())
                plt.savefig(os.path.join(avg_stats_dir, f"{round_id}_std_error_{metric}.png"))
                plt.close()  # Free memory

    print(f"Seed-averaged plots saved successfully at {avg_stats_dir}!")


if __name__ == "__main__":
    # plot_cumulative_points("/home/mila/d/dereck.piche/llm_negotiation/important_outputs/2025-01-12 naive RL with 12 rounds/statistics/alice/alice_stats.jsonl")
    folder = "../scratch/outputs/2025-02-15/08-57-15-fair-bias"
    plot_seed_averaged_stats(folder, ["alice", "bob"])
