import json
import os
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Tuple, Any, Optional
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

def get_mean_statrees(trees: List[Dict]) -> Dict:
    """
    Computes a smart element-wise mean across multiple statrees.
    
    For each path that exists in any of the input trees:
    - If the path contains arrays in multiple trees, computes the element-wise mean
    - Handles arrays of different lengths by taking means of available values at each index
    - If a path exists in only some trees, still computes the mean using available data
    
    Args:
        trees: List of statree dictionaries to compute means from
        
    Returns:
        A new statree where each leaf is the element-wise mean of the corresponding array leaves in the input trees
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
            result[key] = get_mean_statrees(values_at_key)
            
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
                    lst[i] for lst in list_values 
                    if i < len(lst) and lst[i] is not None
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

def update_agent_statistics(input_path, output_file):
    """
    Computes statistics for the current iteration and updates the global statistics file.

    Args:
        input_path (str): Path to the folder containing agent JSON files for the current iteration.
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

def generate_agent_stats_plots(global_stats_path, matplotlib_log_dir, tensorboard_log_dir, wandb_log_dir):
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

def common_plot_statrees(statrees_with_info: List[Tuple[Dict, dict]], output_dir: str, path: str = ""):
    """
    Plot multiple statrees on the same graph for comparison.
    
    Args:
        statrees_with_info: List of tuples, each containing (statree, plot_info)
            where plot_info is a dict with keys:
            - 'color': color to use for this statree's lines
            - 'alpha': transparency level (0-1)
            - 'linewidth': width of the plotted line
            - 'linestyle': style of the line ('-', '--', ':', etc.)
            - 'label': label for the legend
            - 'is_mean': boolean indicating if this is a mean line (for styling)
            - Any other matplotlib Line2D property can be passed
        output_dir: Directory to save the plots
        path: Current path in the statree hierarchy for recursive calls
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # First, find all unique keys across all statrees at this level
    all_keys = set()
    for statree, _ in statrees_with_info:
        all_keys.update(statree.keys())
    
    for key in all_keys:
        new_path = f"{path}/{key}" if path else key
        
        # Collect all dictionaries and lists at this key position
        dicts_at_key = []
        lists_at_key = []
        
        for statree, plot_info in statrees_with_info:
            if key in statree:
                value = statree[key]
                if isinstance(value, dict):
                    dicts_at_key.append((value, plot_info))
                elif isinstance(value, list):
                    lists_at_key.append((value, plot_info))
        
        # If we found dictionaries, recurse into them
        if dicts_at_key:
            common_plot_statrees(dicts_at_key, output_dir, new_path)
        
        # If we found lists, plot them
        if lists_at_key:
            plt.figure(figsize=(10, 6))
            
            # Plot each list with its styling
            for data_list, plot_info in lists_at_key:
                # Extract known parameters
                color = plot_info.get('color', 'blue')
                alpha = plot_info.get('alpha', 1.0)
                linewidth = plot_info.get('linewidth', 1)
                linestyle = plot_info.get('linestyle', '-')
                label = plot_info.get('label', None)
                
                # Skip empty lists
                if not data_list:
                    continue
                
                # Clean up None values
                cleaned_data = [v if v is not None else 0 for v in data_list]
                
                # Create a plot options dictionary for matplotlib
                plot_options = {
                    'color': color, 
                    'alpha': alpha,
                    'linewidth': linewidth,
                    'linestyle': linestyle
                }
                
                # Add label only if provided
                if label:
                    plot_options['label'] = label
                
                # Add any other matplotlib parameters from plot_info
                for k, v in plot_info.items():
                    if k not in ['color', 'alpha', 'linewidth', 'linestyle', 'label', 'is_mean']:
                        plot_options[k] = v
                
                # Plot the data
                plt.plot(cleaned_data, **plot_options)
            
            # Add plot metadata
            plt.title(new_path)
            plt.xlabel("Iterations")
            plt.ylabel(key.replace("_", " ").title())
            
            # Add legend if we have labels
            if any(info.get('label') for _, info in lists_at_key):
                plt.legend()
            
            # Save the plot
            output_filename = os.path.join(output_dir, f"{new_path.replace('/', '_')}.png")
            plt.savefig(output_filename)
            plt.close()


def plot_seed_averaged_stats(root_path, agent_names):
    """
    Plots seed-averaged statistics for given agents.
    
    This function is kept for backwards compatibility but uses the improved implementation.

    Args:
        root_path (str): Path to the directory containing seed data.
        agent_names (list): List of agent names to process.
    """
    seed_dirs = []
    # Identify all seed directories
    for date_dir in os.listdir(root_path):
        date_path = os.path.join(root_path, date_dir)
        if os.path.isdir(date_path):
            seed_dirs.extend([os.path.join(date_path, dir_name) for dir_name in os.listdir(date_path) if dir_name.startswith("seed")])
    
    for agent in agent_names:
        # Create output directory for averaged stats
        avg_stats_dir = os.path.join(root_path, "avg_seed_stats", agent)
        os.makedirs(avg_stats_dir, exist_ok=True)
        
        # Collect paths to all statistic files for this agent
        stat_paths = []
        for seed_dir in seed_dirs:
            stats_file = os.path.join(seed_dir, "statistics", agent, f"{agent}_stats.jsonl")
            if os.path.exists(stats_file):
                stat_paths.append(stats_file)
        
        # Use the improved implementation
        if stat_paths:
            plot_seed_averaged_stats_improved(stat_paths, avg_stats_dir, agent)
    
    print(f"Seed-averaged plots saved successfully at {os.path.join(root_path, 'avg_seed_stats')}!")

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
    
    # Example 3: Computing means across multiple statrees
    # trees = []
    # for path in ["/path/to/tree1.json", "/path/to/tree2.json", "/path/to/tree3.json"]:
    #     with open(path, 'r') as f:
    #         trees.append(json.load(f))
    # 
    # # Compute the mean across all trees
    # mean_tree = get_mean_statrees(trees)
    # 
    # # Save the resulting mean tree
    # with open("/path/to/output/mean_tree.json", 'w') as f:
    #     json.dump(mean_tree, f, indent=2)
    
    # Original example (still works)
    folder = "/home/mila/d/dereck.piche/scratch/finalseeds"
    plot_seed_averaged_stats(folder, ["alice", "bob"])
