#!/usr/bin/env python3
"""
Script to generate multi-color plots from multiple experiment folders without averaging.
"""

import glob
import json
import os
import pathlib
import re
import sys
from typing import Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

from src.utils.log_statistics import common_plot_leafstats


def find_statistic_files(folder_path: str, pattern: str = "**/*.jsonl") -> List[str]:
    """
    Find all statistic files in a folder that match the given pattern.

    Args:
        folder_path: Path to search in
        pattern: Glob pattern for files to find

    Returns:
        List of file paths
    """
    search_path = os.path.join(folder_path, pattern)
    return glob.glob(search_path, recursive=True)


def collect_leafstats(
    input_folders: Union[str, List[str]],
    pattern: str = "**/*.jsonl",
    filter_func: Optional[Callable[[str], bool]] = None,
) -> List[Tuple[Dict, str]]:
    """
    Load statistic trees from JSON files in the given folders.

    Args:
        input_folders: Single folder path or list of folder paths to search in
        pattern: Glob pattern for files to find
        filter_func: Optional function to filter file paths (takes a path, returns bool)

    Returns:
        List of tuples (leafstats, source_path)
    """
    # Handle single folder or list of folders
    if isinstance(input_folders, str):
        input_folders = [input_folders]

    leafstats_with_sources = []

    # Process each input folder
    for folder in input_folders:
        print(f"Processing folder: {folder}")

        # Find all files matching the pattern
        json_files = find_statistic_files(folder, pattern)

        # Apply filter if provided
        if filter_func is not None:
            json_files = [f for f in json_files if filter_func(f)]

        print(f"Found {len(json_files)} JSON files in {folder}")

        # Load leafstats from JSON files
        for file_path in json_files:
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    # Extract a meaningful label from the file path
                    # Try to extract the seed number
                    seed_match = re.search(r"seed_(\d+)", file_path)
                    if seed_match:
                        label = f"Seed {seed_match.group(1)}"
                    else:
                        # Use the filename if we can't find a seed number
                        label = os.path.basename(file_path).split(".")[0]

                    leafstats_with_sources.append((data, label))
            except json.JSONDecodeError:
                print(f"Error: Could not parse JSON in {file_path}")
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")

    return leafstats_with_sources


def plot_multiple_runs(
    input_folders: Union[str, List[str]],
    output_path: Optional[str] = None,
    pattern: str = "**/seed_*/statistics/*/*.jsonl",
):
    """
    Generate plots from multiple runs with different colors and a legend.

    Args:
        input_folders: Folder or list of folders containing statistic data
        output_path: Path to save output plots (defaults to input folder path if single folder)
        pattern: Glob pattern for finding statistic files
    """
    # Handle single folder case for output path
    if isinstance(input_folders, str):
        folders = [input_folders]
    else:
        folders = input_folders

    # Set default output path
    if output_path is None and folders:
        output_path = os.path.join(folders[0], "multi_color_plots")

    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Collect all leafstats from the input folders with their source labels
    leafstats_with_sources = collect_leafstats(input_folders, pattern)

    if not leafstats_with_sources:
        print("No statistic data found in the provided folders!")
        return

    print(f"Collected {len(leafstats_with_sources)} statistic trees")

    # Prepare data for plotting with different colors for each run
    leafstats_with_info = []

    # Use a colormap to get different colors
    # Choose a colormap that gives visually distinct colors
    cmap = plt.cm.get_cmap("tab10", len(leafstats_with_sources))

    # Add each leafstats with a unique color and label
    for i, (leafstats, label) in enumerate(leafstats_with_sources):
        color = cmap(i)
        leafstats_with_info.append(
            (
                leafstats,
                {"color": color, "alpha": 0.8, "linewidth": 1.5, "label": label},
            )
        )

    # Generate plots
    common_plot_leafstats(leafstats_with_info, output_path)

    print(f"Multi-color plots saved to {output_path}")


if __name__ == "__main__":
    # Example usage

    # Example path to experiment folder
    experiment_folder = "/home/mila/d/dereck.piche/scratch/COOPERATIVE"

    agent_name = "bob"

    # For agent's statistics - using a pattern that finds stats files
    pattern = f"**/seed_*/statistics/{agent_name}/{agent_name}_stats.jsonl"

    # Generate multi-color plots
    plot_multiple_runs(
        input_folders=experiment_folder,
        output_path=os.path.join(experiment_folder, f"{agent_name}_multi_color_plots"),
        pattern=pattern,
    )
