from __future__ import annotations

import pickle
from pathlib import Path
from typing import List

from mllm.markov_games.rollout_tree import RolloutTreeRootNode


def find_iteration_folders(global_folder):
    """Find all iteration_* folders within the global folder structure."""
    global_path = Path(global_folder)

    iteration_folders = []

    for item in global_path.glob("iteration_*"):
        if item.is_dir():
            iteration_folders.append(item)

    for seed_dir in global_path.glob("seed_*/"):
        if seed_dir.is_dir():
            for item in seed_dir.glob("iteration_*"):
                if item.is_dir():
                    iteration_folders.append(item)

    return sorted(iteration_folders)


def gather_rollout_trees(iteration_folder):
    """Gather all rollout trees from the iteration folder (.pkl only)."""
    rollout_trees = []
    iteration_path = Path(iteration_folder)
    for item in iteration_path.glob("**/*.rt.pkl"):
        with open(item, "rb") as f:
            data = pickle.load(f)
        rollout_tree = RolloutTreeRootNode.model_validate(data)
        rollout_trees.append(rollout_tree)
    return rollout_trees


def get_rollout_trees(global_folder) -> List[List[RolloutTreeRootNode]]:
    """Get all rollout trees from the global folder."""
    iteration_folders = find_iteration_folders(global_folder)
    rollout_trees = []
    for iteration_folder in iteration_folders:
        rollout_trees.append(gather_rollout_trees(iteration_folder))
    return rollout_trees


import argparse
import glob
import os
import re
import shutil
from pathlib import Path

from mllm.markov_games.gather_and_export_utils import *
from mllm.training.produce_training_stats import render_iteration_trainer_stats


def process_single_folder(
    input_dir,
    output_dir=None,
    per_agent=True,
    include_state_end=False,
    sim_csv=True,
    recursive=False,
):
    """Process a single folder containing PKL rollout tree files (.rt.pkl)."""
    input_path = Path(input_dir)

    # If no output_dir specified, create analysis files in the same input folder
    if output_dir is None:
        output_path = input_path
    else:
        output_path = Path(output_dir)

    pattern = "**/*.rt.pkl" if recursive else "*.rt.pkl"
    files = sorted(input_path.glob(pattern))
    if not files:
        print(f"No PKL rollout trees found in {input_path} (recursive={recursive}).")
        return False

    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Processing folder: {input_path}")
    print(f"Output folder: {output_path}")

    for i, f in enumerate(files, 1):
        print(f"  [{i}/{len(files)}] {f.name}")
        export_rewards_to_csv(
            path=f,
            outdir=output_path,
            first_file=True if i == 1 else False,
        )
        export_chat_logs(
            path=f,
            outdir=output_path,
        )
        export_html_from_rollout_tree(
            path=f,
            outdir=output_path,
            main_only=False,
        )
    # Also render trainer stats from any *.tally.pkl in this iteration folder
    render_iteration_trainer_stats(
        iteration_dir=str(input_path),
        outdir=os.path.join(str(output_path), "trainer_stats.render."),
    )

    return True


def find_iteration_folders(global_folder, from_iteration=0):
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

    return sorted(iteration_folders, key=lambda path: int(path.name.split("_")[-1]))[
        from_iteration:
    ]


def clean_render_artifacts(base_path: Path) -> int:
    """Remove files and directories whose names contain '.render.' under base_path.

    Returns the number of items removed.
    """
    removed_count = 0
    # Ensure path exists
    if not base_path.exists():
        print(f"Path does not exist: {base_path}")
        return 0

    # Traverse all entries under base_path
    for entry in base_path.rglob("*"):
        try:
            if ".render." in entry.name:
                if entry.is_dir():
                    shutil.rmtree(entry, ignore_errors=True)
                    print(f"Removed directory: {entry}")
                    removed_count += 1
                elif entry.is_file() or entry.is_symlink():
                    try:
                        entry.unlink()
                        print(f"Removed file: {entry}")
                        removed_count += 1
                    except FileNotFoundError:
                        # Already gone
                        pass
        except Exception as e:
            print(f"Failed to inspect/remove {entry}: {e}")

    # Also check the base_path itself
    if ".render." in base_path.name:
        try:
            if base_path.is_dir():
                shutil.rmtree(base_path, ignore_errors=True)
                print(f"Removed directory: {base_path}")
                removed_count += 1
            elif base_path.is_file() or base_path.is_symlink():
                base_path.unlink()
                print(f"Removed file: {base_path}")
                removed_count += 1
        except Exception as e:
            print(f"Failed to remove base path {base_path}: {e}")

    return removed_count


def main():
    parser = argparse.ArgumentParser(
        description="Process negotiation game rollout files for analysis"
    )
    # Positional path acts as --global-folder for convenience
    parser.add_argument(
        "positional_global_folder",
        nargs="?",
        help="Global folder containing iteration_* subdirectories (positional; same as --global-folder)",
    )
    parser.add_argument(
        "path",
        nargs="?",
        help="Positional path to the global experiment folder (equivalent to --global-folder PATH)",
    )
    parser.add_argument(
        "--folders", nargs="+", help="List of specific folders to process"
    )
    parser.add_argument(
        "--global-folder",
        default=".",
        help="Global folder containing iteration_* subdirectories (default: current directory)",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory (if not specified, files are created in input folders)",
    )
    parser.add_argument(
        "--per-agent",
        action="store_true",
        default=True,
        help="Also write per-agent transcripts (default: True)",
    )
    parser.add_argument(
        "--no-per-agent",
        action="store_false",
        dest="per_agent",
        help="Don't write per-agent transcripts",
    )
    parser.add_argument(
        "--include-state-end",
        action="store_true",
        default=False,
        help="Annotate <STATE_END> on lines (default: False)",
    )
    parser.add_argument(
        "--sim-csv",
        action="store_true",
        default=True,
        help="Export simulation infos to CSV (default: True)",
    )
    parser.add_argument(
        "--no-sim-csv",
        action="store_false",
        dest="sim_csv",
        help="Don't export simulation infos to CSV",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=False,
        help="Search subfolders for JSON files (default: False)",
    )
    parser.add_argument(
        "--from-iteration",
        type=int,
        default=0,
        help="Start processing from a specific iteration (default: 0)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove all files/directories with '.render.' in their name under the given path(s) and exit",
    )

    args = parser.parse_args()

    # Positional path, if provided, takes precedence over --global-folder
    effective_global_folder = args.path if args.path else args.global_folder

    # Handle cleaning mode first
    if args.clean:
        target_paths = []
        if effective_global_folder:
            target_paths.append(Path(effective_global_folder))
        if args.folders:
            target_paths.extend([Path(p) for p in args.folders])
        if not target_paths:
            target_paths = [Path(".")]

        total_removed = 0
        for base in target_paths:
            print(f"Cleaning under: {base}")
            total_removed += clean_render_artifacts(base)

        print(
            f"Cleaning complete. Removed {total_removed} item(s) containing '.render.'"
        )
        return

    folders_to_process = []

    if effective_global_folder:
        # Find all iteration_* folders in the global folder
        iteration_folders = find_iteration_folders(
            effective_global_folder, args.from_iteration
        )
        if not iteration_folders:
            print(f"No iteration_* folders found in {effective_global_folder}")
            return
        folders_to_process.extend(iteration_folders)
        print(
            f"Found {len(iteration_folders)} iteration folders in {effective_global_folder}"
        )

    if args.folders:
        # Add specified folders
        folders_to_process.extend([Path(f) for f in args.folders])

    if not folders_to_process:
        print("No folders to process.")
        return

    # Process each folder
    successful_count = 0
    for folder in folders_to_process:
        if process_single_folder(
            folder,
            args.output_dir,
            per_agent=args.per_agent,
            include_state_end=args.include_state_end,
            sim_csv=args.sim_csv,
            recursive=args.recursive,
        ):
            successful_count += 1

    print(
        f"\nProcessing complete. Successfully processed {successful_count}/{len(folders_to_process)} folders."
    )


if __name__ == "__main__":
    main()
