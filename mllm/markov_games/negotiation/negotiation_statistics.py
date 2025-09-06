from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from statistics_runner import run_stats_functional

from mllm.markov_games.rollout_tree import SimulationStepLog


def metric_greed(sl: SimulationStepLog) -> Dict[str, float] | None:
    info = sl.info or {}
    if not info or not info.get("is_last_timestep_in_round"):
        return None
    quantities = info.get("quantities") or {}
    denom = float(quantities.get("coins", 1.0)) or 1.0
    splits = info.get("splits") or {}
    out: Dict[str, float] = {}
    for aid, split in splits.items():
        try:
            out[str(aid)] = float(split["items_given_to_self"]["coins"]) / denom
        except Exception:
            continue
    return out


def metric_efficiency(sl: SimulationStepLog) -> Dict[str, float] | None:
    info = sl.info or {}
    if not info or not info.get("is_last_timestep_in_round"):
        return None
    quantities = info.get("quantities") or {}
    denom = float(quantities.get("coins", 1.0)) or 1.0
    values = info.get("values") or {}
    if not values:
        return None
    try:
        max_val = max(float(v) for v in values.values())
    except Exception:
        return None
    if not denom or not max_val:
        return None
    achieved = sum(float(v) for v in (sl.rewards or {}).values())
    max_reward = denom * max_val
    if not max_reward:
        return None
    # Efficiency is a global metric; emit same value for a special key "all"
    return {"all": achieved / max_reward}


def main():
    parser = argparse.ArgumentParser(description="Compute negotiation statistics fast")
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
        "avg_greed": metric_greed,
        "avg_efficiency": metric_efficiency,
    }
    out = run_stats_functional(
        Path(args.data_root),
        "negotiation",
        metrics,
        args.outfile,
        output_format=args.format,
    )
    print(str(out))


if __name__ == "__main__":
    main()
