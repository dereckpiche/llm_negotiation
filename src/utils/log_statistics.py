
import json
import os
from statistics import mean
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, Union, List

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

    
def generate_player_stats_plots(global_stats_path, plot_folder, tensorboard_log_dir):
    """
    Visualizes the global statistics by generating plots and logging to TensorBoard.

    Args:
        global_stats_path (str): Path to the global statistics JSON file.
        plot_folder (str): Folder to save the plots.
        tensorboard_log_dir (str): Directory to save TensorBoard logs.
    """
    with open(global_stats_path, 'r') as f:
        global_stats = json.load(f)

    # Plot statistics and save to folder
    plot_statree(global_stats, plot_folder)

    # Log statistics to TensorBoard
    writer = SummaryWriter(tensorboard_log_dir)
    tb_statree(global_stats, writer)
    writer.close()

def plot_cumulative_points(json_path):
    """
    Generates a plot of cumulative points for Alice and Bob over iterations 
    from a JSON file and saves it in the same directory as the input JSON.

    Args:
        json_path (str): Path to the JSON file containing the statistics.
    """
    try:
        with open(json_path, 'r') as file:
            statistics = json.load(file)
    except FileNotFoundError:
        print(f"Error: File not found at {json_path}. Please check the file path and try again.")
        return

    nb_epochs = len(statistics["round_0"]["self_points"])
    alice_cumu_points = np.zeros(nb_epochs)
    bob_cumu_points = np.zeros(nb_epochs)

    for round_key in statistics.keys():
        alice_cumu_points += np.array(statistics[round_key]["self_points"])
        bob_cumu_points += np.array(statistics[round_key]["other_points"])

    iterations = np.arange(1, nb_epochs + 1)
    output_dir = os.path.dirname(json_path)
    plot_filename = "average_cumulative_points_plot.png"
    output_path = os.path.join(output_dir, plot_filename)

    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, alice_cumu_points, label="Alice's Avg Cumulative Points")
    plt.plot(iterations, bob_cumu_points, label="Bob's Avg Cumulative Points")
    plt.xlabel("Iterations")
    plt.ylabel("Average Cumulative Points")
    plt.title("Average Cumulative Points Over Iterations")
    plt.legend()
    plt.grid()
    plt.savefig(output_path)

    print(f"Plot saved successfully at: {output_path}")

if __name__ == "__main__": 
    plot_cumulative_points("/home/mila/d/dereck.piche/llm_negotiation/important_outputs/2025-01-12 naive RL with 12 rounds/statistics/alice/alice_stats.jsonl")
