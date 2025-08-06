from pathlib import Path

from mllm.markov_games.render_utils import *
from mllm.markov_games.rollout_tree import *


def find_iteration_folders(global_folder):
    """Find all iteration_* folders within the global folder structure."""
    global_path = Path(global_folder)

    # Look for iteration_* folders in all subdirectories
    iteration_folders = []

    # Search in the global folder itself
    for item in global_path.glob("iteration_*"):
        if item.is_dir():
            iteration_folders.append(item)

    # Search in seed_* subdirectories
    for seed_dir in global_path.glob("seed_*/"):
        if seed_dir.is_dir():
            for item in seed_dir.glob("iteration_*"):
                if item.is_dir():
                    iteration_folders.append(item)

    return sorted(iteration_folders)


def gather_rollout_trees(iteration_folder):
    """Gather all rollout trees from the iteration folder."""
    rollout_trees = []
    iteration_path = Path(iteration_folder)
    for item in iteration_path.glob("**/*.json"):
        if item.is_file() and re.match(
            r"(rollout_tree\.json|mgid:\d+_rollout_tree\.json)$", item.name
        ):
            rollout_tree = RolloutTreeRootNode.model_validate_json(item.read_text())
            rollout_trees.append(rollout_tree)
    return rollout_trees


def get_rollout_trees(global_folder) -> list[list[RolloutTreeRootNode]]:
    """Get all rollout trees from the global folder."""
    iteration_folders = find_iteration_folders(global_folder)
    rollout_trees = []
    for iteration_folder in iteration_folders:
        rollout_trees.append(gather_rollout_trees(iteration_folder))
    return rollout_trees
