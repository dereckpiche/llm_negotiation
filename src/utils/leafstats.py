import json
import os
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# import wandb
# from comet_ml import Experiment

# experiment = Experiment(
#     api_key="IvI06nn59lLap4y0JRrwlTViy",
#     project_name="llm_negotiation",
#     log_env_gpu=True,
#     log_env_cpu=True
# )


def append_leafstats(tree1: Dict, tree2: Dict):
    """
    Append each corresponding leaf of tree2 to tree1.
    """
    for key, value in tree2.items():
        if key not in tree1:
            tree1[key] = value
        elif isinstance(value, dict):
            append_leafstats(tree1[key], value)
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


def get_mean_leafstats(tree: Dict) -> Dict:
    """
    scores a leafstats where each leaf is replaced by the mean of the leaf values.
    """
    result = {}
    for key, value in tree.items():
        if isinstance(value, dict):
            result[key] = get_mean_leafstats(value)
        elif isinstance(value, list):
            cleaned_values = [v for v in value if v is not None]
            result[key] = np.mean(cleaned_values) if cleaned_values else None
        else:
            result[key] = value
    return result


def get_mean_leafstats(trees: List[Dict]) -> Dict:
    """
    Computes a smart element-wise mean across multiple leafstats.

    For each path that exists in any of the input trees:
    - If the path contains arrays in multiple trees, computes the element-wise mean
    - Handles arrays of different lengths by taking means of available values at each index
    - If a path exists in only some trees, still computes the mean using available data

    Args:
        trees: List of leafstats dictionaries to compute means from

    Returns:
        A new leafstats where each leaf is the element-wise mean of the corresponding array leaves in the input trees
    """
    if not trees:
        return {}

    if len(trees) == 1:
        return trees[0]

    # Collect all possible keys at this level
    all_keys = set()
    for tree in trees:
        all_keys.update(tree.keys())

    result = {}

    for key in all_keys:
        # Collect all values at this key position
        values_at_key = []

        for tree in trees:
            if key in tree:
                values_at_key.append(tree[key])

        # If all values are dictionaries, recursively compute means
        if all(isinstance(v, dict) for v in values_at_key):
            result[key] = get_mean_leafstats(values_at_key)

        # If any value is a list, compute element-wise means
        elif any(isinstance(v, list) for v in values_at_key):
            # First, convert non-list values to lists (singleton)
            list_values = []
            for v in values_at_key:
                if isinstance(v, list):
                    list_values.append(v)
                else:
                    list_values.append([v])

            # Find the maximum length among all lists
            max_length = max(len(lst) for lst in list_values)

            # Initialize result array
            mean_array = []

            # Compute element-wise means
            for i in range(max_length):
                # Collect values at this index that are not None
                values_at_index = [
                    lst[i] for lst in list_values if i < len(lst) and lst[i] is not None
                ]

                # If we have valid values, compute mean; otherwise, use None
                if values_at_index:
                    mean_array.append(np.mean(values_at_index))
                else:
                    mean_array.append(None)

            result[key] = mean_array

        # If all values are scalars (not dict or list), compute their mean
        else:
            # Filter out None values
            non_none_values = [v for v in values_at_key if v is not None]

            if non_none_values:
                result[key] = np.mean(non_none_values)
            else:
                result[key] = None

    return result


def get_var_leafstats(tree: Dict) -> Dict:
    """
    scores a leafstats where each leaf is replaced by the variance of the leaf values.
    """
    result = {}
    for key, value in tree.items():
        if isinstance(value, dict):
            result[key] = get_var_leafstats(value)
        elif isinstance(value, list):
            cleaned_values = [v for v in value if v is not None]
            result[key] = np.var(cleaned_values) if cleaned_values else None
        else:
            result[key] = 0  # Single value has variance 0
    return result


def plot_leafstats(tree: Dict, folder: str, path: str = ""):
    """
    Plots the leaves of the leafstats and saves them to the specified folder.
    """
    os.makedirs(folder, exist_ok=True)
    for key, value in tree.items():
        new_path = f"{path}_{key}"
        if isinstance(value, dict):
            plot_leafstats(value, folder, new_path)
        elif isinstance(value, list):
            plt.figure()
            plt.plot(value)
            plt.title(new_path)
            plt.savefig(os.path.join(folder, f"{new_path.replace('/', '_')}.png"))
            plt.close()


def plot_EMA_leafstats(tree: Dict, folder: str, path: str = "", alpha: float = 0.1):
    """
    Plots the exponential moving average of the leaves of the leafstats and saves them to the specified folder.
    """
    os.makedirs(folder, exist_ok=True)
    for key, value in tree.items():
        new_path = f"{path}/{key}" if path else key
        if isinstance(value, dict):
            plot_EMA_leafstats(value, folder, new_path, alpha)
        elif isinstance(value, list):
            value = np.array(value)
            nb_elements = len(value)
            coefficients = (1 - alpha) ** np.arange(nb_elements, 0, -1)
            value = np.cumsum(value * coefficients)
            value /= (1 - alpha) ** np.arange(nb_elements - 1, -1, -1)  # renormalize
            out_path = f"EMA_alpha_{alpha}_{path}_{key}"
            out_path = os.path.join(folder, f"{out_path.replace('/', '_')}.png")
            # import pdb; pdb.set_trace()
            plt.figure()
            plt.plot(value)
            plt.title(new_path)
            plt.savefig(out_path)
            plt.close()


def plot_SMA_leafstats(tree: Dict, folder: str, path: str = "", window: int = 33):
    """
    Plots the simple moving average of the leaves of the leafstats and saves them to the specified folder.
    """
    os.makedirs(folder, exist_ok=True)
    for key, value in tree.items():
        new_path = f"{path}/{key}" if path else key
        if isinstance(value, dict):
            plot_SMA_leafstats(value, folder, new_path, window)
        elif isinstance(value, list):
            value = np.array(value)
            nb_elements = len(value)

            assert window % 2 == 1  # Even numbers are annoying for centered windows
            value = np.convolve(v=value, a=np.ones(window) / window, mode="same")
            # Adjust out of window for start and finish
            value[: window // 2] *= window / np.arange(window // 2 + 1, window)
            value[-window // 2 + 1 :] *= window / np.arange(window - 1, window // 2, -1)

            out_path = f"SMA_window_{window}_{path}_{key}"
            out_path = os.path.join(folder, f"{out_path.replace('/', '_')}.png")
            plt.figure()
            plt.plot(value)
            plt.title(new_path)
            plt.savefig(out_path)
            plt.close()


def save_leafstats(tree: Dict, folder: str, path: str = ""):
    """
    Saves the leaves of the leafstats to the specified folder.
    """
    os.makedirs(folder, exist_ok=True)
    for key, value in tree.items():
        new_path = f"{path}/{key}" if path else key
        if isinstance(value, dict):
            save_leafstats(value, folder, new_path)
        elif (
            isinstance(value, list)
            or isinstance(value, np.ndarray)
            or isinstance(value, float)
            or isinstance(value, int)
        ):
            with open(
                os.path.join(folder, f"{new_path.replace('/', '_')}.json"), "w"
            ) as f:
                json.dump(value, f, indent=4)


def tb_leafstats(tree: Dict, writer, path: str = ""):
    """
    Logs the leaves of the leafstats to TensorBoard.
    """
    for key, value in tree.items():
        new_path = f"{path}/{key}" if path else key
        if isinstance(value, dict):
            tb_leafstats(value, writer, new_path)
        elif isinstance(value, list):
            for i, v in enumerate(value):
                if v is not None:
                    writer.add_scalar(new_path, v, i)


def update_agent_statistics(input_path, output_file):
    """
    Computes statistics for the current iteration and updates the global statistics file.

    Args:
        input_path (str): Path to the folder containing agent JSON files for the current iteration.
        output_file (str): Path to the JSON file where statistics are stored.
    """

    # Build leafstats by appending each dict from JSON files in "input_path" folder
    leafstats = {}
    for filename in os.listdir(input_path):
        if filename.endswith(".json"):
            with open(os.path.join(input_path, filename), "r") as f:
                data = json.load(f)
                append_leafstats(leafstats, data)
    # Get epoch mean leafstats
    mean_leafstats = get_mean_leafstats(leafstats)

    # Add mean leafstats to global stats file
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            global_stats = json.load(f)
    else:
        global_stats = {}

    append_leafstats(global_stats, mean_leafstats)

    with open(output_file, "w") as f:
        json.dump(global_stats, f, indent=4)


def generate_agent_stats_plots(
    global_stats_path, matplotlib_log_dir, tensorboard_log_dir, wandb_log_dir
):
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

    with open(global_stats_path, "r") as f:
        global_stats = json.load(f)

    plot_leafstats(global_stats, folder=matplotlib_log_dir)

    # Log statistics to TensorBoard
    writer = SummaryWriter(tensorboard_log_dir)
    tb_leafstats(global_stats, writer)
    writer.close()

    # wb_leafstats(global_stats)


if __name__ == "__main__":
    # Example usage of the improved plotting functions

    # Example 1: Plot seed-averaged statistics for a single agent
    # plot_seed_averaged_stats_improved(
    #     stat_paths=["/path/to/seed1/stats", "/path/to/seed2/stats"],
    #     output_dir="/path/to/output",
    #     agent_name="alice"
    # )

    # Example 2: Compare multiple experiments and show grand mean
    # plot_multi_experiment_comparison(
    #     experiment_dirs=[
    #         ("/path/to/experiment1", "Baseline"),
    #         ("/path/to/experiment2", "Improved Method"),
    #         ("/path/to/experiment3", "Advanced Technique")
    #     ],
    #     output_dir="/path/to/comparison_output",
    #     agent_name="alice",
    #     show_grand_mean=True
    # )

    # Example 3: Computing means across multiple leafstats
    # trees = []
    # for path in ["/path/to/tree1.json", "/path/to/tree2.json", "/path/to/tree3.json"]:
    #     with open(path, 'r') as f:
    #         trees.append(json.load(f))
    #
    # # Compute the mean across all trees
    # mean_tree = get_mean_leafstats(trees)
    #
    # # Save the resulting mean tree
    # with open("/path/to/output/mean_tree.json", 'w') as f:
    #     json.dump(mean_tree, f, indent=2)

    # Original example (still works)
    folder = "/home/mila/d/dereck.piche/scratch/finalseeds"
    plot_seed_averaged_stats(folder, ["alice", "bob"])
