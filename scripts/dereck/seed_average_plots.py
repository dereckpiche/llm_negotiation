#!/usr/bin/env python3
"""
Script to generate seed-averaged plots from multiple experiment folders.
"""

import os
import json
import glob
import re
from typing import Union, List, Optional, Dict, Tuple, Callable

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

from src.utils.log_statistics import get_mean_statrees, common_plot_statrees


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


def collect_statrees(
    input_folders: Union[str, List[str]], 
    pattern: str = "**/*.jsonl",
    filter_func: Optional[Callable[[str], bool]] = None
) -> List[Dict]:
    """
    Load statistic trees from JSON files in the given folders.
    
    Args:
        input_folders: Single folder path or list of folder paths to search in
        pattern: Glob pattern for files to find
        filter_func: Optional function to filter file paths (takes a path, returns bool)
    
    Returns:
        List of loaded statrees
    """
    # Handle single folder or list of folders
    if isinstance(input_folders, str):
        input_folders = [input_folders]
    
    statrees = []
    
    # Process each input folder
    for folder in input_folders:
        print(f"Processing folder: {folder}")
        
        # Find all files matching the pattern
        json_files = find_statistic_files(folder, pattern)
        
        # Apply filter if provided
        if filter_func is not None:
            json_files = [f for f in json_files if filter_func(f)]
        
        print(f"Found {len(json_files)} JSON files in {folder}")
        
        # Load statrees from JSON files
        for file_path in json_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    statrees.append(data)
            except json.JSONDecodeError:
                print(f"Error: Could not parse JSON in {file_path}")
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
    
    return statrees


def plot_multipled_seeds(
    input_folders: Union[str, List[str]], 
    output_path: Optional[str] = None,
    pattern: str = "**/seed_*/statistics/*/*.jsonl"
):
    """
    Generate plots from multiple seed folders, with individual runs in light blue
    and the mean in dark blue.
    
    Args:
        input_folders: Folder or list of folders containing seed statistic data
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
        output_path = os.path.join(folders[0], "seed_averaged_plots")
    
    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)
    
    # Collect all statrees from the input folders
    all_statrees = collect_statrees(input_folders, pattern)
    
    if not all_statrees:
        print("No statistic data found in the provided folders!")
        return
    
    print(f"Collected {len(all_statrees)} statistic trees")
    
    # Compute the mean across all statrees
    mean_statree = get_mean_statrees(all_statrees)
    
    # Prepare data for plotting
    statrees_with_info = []
    
    # Add individual statrees with light blue styling
    for statree in all_statrees:
        statrees_with_info.append((
            statree,
            {
                'color': 'lightblue',
                'alpha': 0.5,
                'linewidth': 1
            }
        ))
    
    # Add mean statree with dark blue styling
    statrees_with_info.append((
        mean_statree,
        {
            'color': 'darkblue',
            'alpha': 1.0,
            'linewidth': 2,
            'label': 'Average'
        }
    ))
    
    # Generate plots
    common_plot_statrees(statrees_with_info, output_path)
    
    # Save the mean statree as JSON
    mean_output_path = os.path.join(output_path, "mean_statree.json")
    with open(mean_output_path, 'w') as f:
        json.dump(mean_statree, f, indent=2)
    
    print(f"Plots and mean statree saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    
    # Example path to experiment folder
    experiment_folder = "/home/mila/d/dereck.piche/scratch/finalseeds"

    agent_name = "bob"
    
    # For alice's statistics - using a pattern that finds alice's stats files
    pattern = f"**/seed_*/statistics/{agent_name}/{agent_name}_stats.jsonl"
    
    # Generate plots for alice's statistics
    plot_multipled_seeds(
        input_folders=experiment_folder,
        output_path=os.path.join(experiment_folder, f"{agent_name}_multi_seed_plots"),
        pattern=pattern
    )
    
    # For bob's statistics - using a pattern that finds bob's stats files
    # bob_pattern = "**/seed_*/statistics/bob/bob_stats.jsonl"
    # plot_multipled_seeds(
    #     input_folders=experiment_folder,
    #     output_path=os.path.join(experiment_folder, "bob_plots"),
    #     pattern=bob_pattern
    # )
