import argparse
import glob
import os
from pathlib import Path

from mllm.markov_games.gather_and_export_utils import *


def process_single_folder(
    input_dir,
    output_dir=None,
    per_agent=True,
    include_state_end=False,
    sim_csv=True,
    recursive=False,
):
    """Process a single folder containing JSON rollout files."""
    input_path = Path(input_dir)

    # If no output_dir specified, create analysis files in the same input folder
    if output_dir is None:
        output_path = input_path
    else:
        output_path = Path(output_dir)

    pattern = "**/*.json" if recursive else "*.json"
    files = sorted(input_path.glob(pattern))
    if not files:
        print(f"No JSON files found in {input_path} (recursive={recursive}).")
        return False

    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Processing folder: {input_path}")
    print(f"Output folder: {output_path}")

    for i, f in enumerate(files, 1):
        print(f"  [{i}/{len(files)}] {f.name}")
        try:
            export_chat_logs(
                path=f,
                outdir=output_path,
            )
            export_html_from_rollout_tree(
                path=f,
                outdir=output_path,
                main_only=False,
            )
        except Exception as e:
            print(f"  !! Error in {f}: {e}")

    return True


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


def main():
    parser = argparse.ArgumentParser(
        description="Process negotiation game rollout files for analysis"
    )
    parser.add_argument(
        "--folders", nargs="+", help="List of specific folders to process"
    )
    parser.add_argument(
        "--global_folder", help="Global folder containing iteration_* subdirectories"
    )
    parser.add_argument(
        "--output_dir",
        help="Output directory (if not specified, files are created in input folders)",
    )
    parser.add_argument(
        "--per_agent",
        action="store_true",
        default=True,
        help="Also write per-agent transcripts (default: True)",
    )
    parser.add_argument(
        "--no_per_agent",
        action="store_false",
        dest="per_agent",
        help="Don't write per-agent transcripts",
    )
    parser.add_argument(
        "--include_state_end",
        action="store_true",
        default=False,
        help="Annotate <STATE_END> on lines (default: False)",
    )
    parser.add_argument(
        "--sim_csv",
        action="store_true",
        default=True,
        help="Export simulation infos to CSV (default: True)",
    )
    parser.add_argument(
        "--no_sim_csv",
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

    args = parser.parse_args()

    # Require at least one input method
    if not args.folders and not args.global_folder:
        parser.error("Must specify either --folders or --global_folder")
        return

    folders_to_process = []

    if args.global_folder:
        # Find all iteration_* folders in the global folder
        iteration_folders = find_iteration_folders(args.global_folder)
        if not iteration_folders:
            print(f"No iteration_* folders found in {args.global_folder}")
            return
        folders_to_process.extend(iteration_folders)
        print(
            f"Found {len(iteration_folders)} iteration folders in {args.global_folder}"
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
