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
