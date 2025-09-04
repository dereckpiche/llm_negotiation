from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from statistics_runner import run_stats_functional

from mllm.markov_games.rollout_tree import SimulationStepLog


def metric_avg_reward(sl: SimulationStepLog) -> Dict[str, float]:
    # One value per agent at each step
    return {aid: float(v) for aid, v in (sl.rewards or {}).items()}


def main():
    parser = argparse.ArgumentParser(description="Compute IPD statistics fast")
    parser.add_argument(
        "data_root", type=str, help="Path to folder containing iteration_* subfolders"
    )
    parser.add_argument(
        "--outfile",
        type=str,
        default=None,
        help="Optional output filename inside statistics/",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="json",
        choices=["json", "jsonl"],
        help="Output format: json (dict of lists) or jsonl",
    )
    args = parser.parse_args()

    metrics = {
        "avg_reward": metric_avg_reward,
    }
    out = run_stats_functional(
        Path(args.data_root), "ipd", metrics, args.outfile, output_format=args.format
    )
    print(str(out))


if __name__ == "__main__":
    main()
