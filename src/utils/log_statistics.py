
import json
import os
from statistics import mean
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
from torch.utils.tensorboard import SummaryWriter
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, Union, List
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
    Returns a statree where each leaf is replaced by the mean of the leaf values.
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
    Returns a statree where each leaf is replaced by the variance of the leaf values.
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



if __name__ == "__main__":
    plot_cumulative_points("/home/mila/d/dereck.piche/llm_negotiation/important_outputs/2025-01-12 naive RL with 12 rounds/statistics/alice/alice_stats.jsonl")
