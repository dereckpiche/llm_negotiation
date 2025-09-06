from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    tqdm = None

from mllm.markov_games.rollout_tree import (
    RolloutTreeBranchNode,
    RolloutTreeNode,
    RolloutTreeRootNode,
    SimulationStepLog,
)
from render import find_iteration_folders


def _iterate_main_nodes_fast(root: dict) -> Iterator[dict]:
    """Iterate main trajectory nodes from a raw dict-serialized rollout tree."""
    current = root.get("child") if isinstance(root, dict) else None
    while current is not None:
        if isinstance(current, dict) and ("step_log" in current):
            yield current
            current = current.get("child")
        elif isinstance(current, dict) and ("main_child" in current):
            current = current.get("main_child")
        else:
            break


class _SlView:
    __slots__ = ("rewards", "info")

    def __init__(self, rewards, info):
        self.rewards = rewards
        self.info = info


def iterate_main_simulation_logs_fast(root: dict) -> Iterator[_SlView]:
    for node in _iterate_main_nodes_fast(root):
        sl = (node.get("step_log") or {}).get("simulation_step_log") or {}
        yield _SlView(sl.get("rewards", {}), sl.get("info"))


def metric_avg_reward(sl: _SlView) -> Optional[Dict[str, float]]:
    """Per-step reward by agent; averaged later across steps and rollouts."""
    rewards = getattr(sl, "rewards", None)
    if not isinstance(rewards, dict):
        return None
    out: Dict[str, float] = {}
    for aid, val in rewards.items():
        try:
            out[str(aid)] = float(val)
        except Exception:
            continue
    return out if out else None


avg_reward = metric_avg_reward


def stream_rollout_files(iteration_folder: Path) -> Iterator[Path]:
    for p in iteration_folder.rglob("*.rt.pkl"):
        if p.is_file():
            yield p


def load_root(path: Path) -> dict:
    """Load rollout tree as raw dict without Pydantic validation (fast)."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    # data is expected to be a dict representing the tree
    return data


def _compute_record_for_file(
    pkl_path: str,
    iteration_name: str,
    metrics_items: List[Tuple[str, Callable[[Any], Optional[Dict[str, float]]]]],
):
    root = load_root(Path(pkl_path))
    agg: Dict[str, Dict[str, List[float]]] = {m: {} for m, _ in metrics_items}
    for sl in iterate_main_simulation_logs_fast(root):
        for mname, fn in metrics_items:
            vals = None
            try:
                vals = fn(sl)
            except Exception:
                vals = None
            if not vals:
                continue
            for aid, v in vals.items():
                if v is None:
                    continue
                lst = agg[mname].setdefault(str(aid), [])
                try:
                    lst.append(float(v))
                except Exception:
                    continue

    # finalize averages per metric/agent for this rollout
    result: Dict[str, Dict[str, float]] = {}
    for mname, _ in metrics_items:
        result[mname] = {}
        for aid, vals in agg[mname].items():
            result[mname][aid] = (sum(vals) / len(vals)) if vals else None

    mgid = root.get("id") if isinstance(root, dict) else None
    crn_id = root.get("crn_id") if isinstance(root, dict) else None
    return {
        "mgid": mgid,
        "crn_id": crn_id,
        "iteration": iteration_name,
        "stats": result,
    }


def run_stats_functional(
    data_root: Path,
    game_name: str,
    metrics: Dict[str, Callable[[SimulationStepLog], Optional[Dict[str, float]]]],
    jobs: Optional[int] = None,
    from_iteration: int = 0,
    last_iteration: Optional[int] = None,
) -> Path:
    data_root = Path(data_root)
    # Write a single iteration-averaged stats file at the root
    outfile = data_root / "statistics.json"
    if outfile.exists():
        outfile.unlink()

    # Ensure deterministic ordering of iterations (numeric sort) and apply range filters
    all_iteration_folders = list(find_iteration_folders(str(data_root)))
    parsed: List[Tuple[int, str]] = []
    for p in all_iteration_folders:
        try:
            idx = int(Path(p).name.split("_")[-1])
        except Exception:
            continue
        if idx < from_iteration:
            continue
        if last_iteration is not None and idx > last_iteration:
            continue
        parsed.append((idx, str(p)))
    parsed.sort(key=lambda t: t[0])
    iteration_folders = [p for (_idx, p) in parsed]

    def finalize_rollout(
        agg: Dict[str, Dict[str, List[float]]]
    ) -> Dict[str, Dict[str, float]]:
        result: Dict[str, Dict[str, float]] = {}
        for mname, agent_values in agg.items():
            result[mname] = {}
            for aid, vals in agent_values.items():
                if not vals:
                    result[mname][aid] = None
                else:
                    result[mname][aid] = sum(vals) / len(vals)
        return result

    records: List[Dict[str, Any]] = []

    # Prepare tasks across all iterations to use a single process pool
    metrics_items = list(metrics.items())
    max_workers = jobs or max(1, (os.cpu_count() or 2))
    tasks: List[Tuple[str, str]] = []  # (pkl_path, iteration_name)
    for iteration_folder in iteration_folders:
        iteration_name = Path(iteration_folder).name
        # Collect and sort rollout files deterministically
        files = sorted(
            (p for p in stream_rollout_files(Path(iteration_folder))),
            key=lambda x: str(x),
        )
        for p in files:
            tasks.append((str(p), iteration_name))

    # Sort tasks deterministically by (iteration_name, path)
    tasks.sort(key=lambda t: (t[1], t[0]))

    # Compute all rollout records (no intermediate files)
    total_rollouts = len(tasks)
    pbar = (
        tqdm(total=total_rollouts, desc="Rollouts", dynamic_ncols=True)
        if (tqdm and hasattr(sys.stderr, "isatty") and sys.stderr.isatty())
        else None
    )
    processed = 0
    if max_workers > 1 and total_rollouts > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures_map = {}
            for idx, (p, it) in enumerate(tasks):
                fut = ex.submit(_compute_record_for_file, p, it, metrics_items)
                futures_map[fut] = idx
            results: List[Optional[Dict[str, Any]]] = [None] * total_rollouts
            for fut in as_completed(futures_map.keys()):
                idx = futures_map[fut]
                results[idx] = fut.result()
                if pbar is not None:
                    pbar.update(1)
                else:
                    processed += 1
                    print(f"Rollouts: {processed}/{total_rollouts}", flush=True)
        records = [rec for rec in results if rec is not None]
    else:
        for p, it in tasks:
            rec = _compute_record_for_file(p, it, metrics_items)
            records.append(rec)
            if pbar is not None:
                pbar.update(1)
            else:
                processed += 1
                print(f"Rollouts: {processed}/{total_rollouts}", flush=True)
    if pbar is not None:
        pbar.close()

    # CSV writing removed for performance; keeping only iteration averages output

    # Write per-iteration averages (across mgids) as JSON
    def write_iteration_avgs(records_list: List[Dict[str, Any]]):
        # Group by iteration and remember iteration order
        by_iter: Dict[str, List[Dict[str, Any]]] = {}
        iteration_order: List[str] = []
        for r in records_list:
            it = str(r.get("iteration"))
            if it not in by_iter:
                by_iter[it] = []
                iteration_order.append(it)
            by_iter[it].append(r)

        # Enforce deterministic iteration order (lexicographic suits iteration_000 format)
        iteration_order = sorted(iteration_order)

        # Determine metric names and agent keys
        metric_names: set[str] = set()
        agent_keys_per_metric: Dict[str, set[str]] = {}
        for r in records_list:
            stats = r.get("stats", {}) or {}
            for mname, val in stats.items():
                metric_names.add(mname)
                if isinstance(val, dict):
                    s = agent_keys_per_metric.setdefault(mname, set())
                    for ak in val.keys():
                        s.add(str(ak))

        # Compute averages per iteration (iteration -> metric -> agent -> value)
        iter_avgs: Dict[str, Dict[str, Dict[str, float]]] = {}
        for it_name, recs in by_iter.items():
            iter_avgs[it_name] = {}
            for mname in metric_names:
                iter_avgs[it_name][mname] = {}
                agent_cols = sorted(agent_keys_per_metric.get(mname, set()))
                for ak in agent_cols:
                    vals: List[float] = []
                    for r in recs:
                        stats = (r.get("stats") or {}).get(mname)
                        if (
                            isinstance(stats, dict)
                            and ak in stats
                            and stats[ak] is not None
                        ):
                            try:
                                vals.append(float(stats[ak]))
                            except Exception:
                                pass
                    iter_avgs[it_name][mname][ak] = (
                        (sum(vals) / len(vals)) if vals else None
                    )

        # Reformat to metric-first with agent arrays over iterations: stats[metric][agent] = [v_it0, v_it1, ...]
        metric_first: Dict[str, Dict[str, List[float]]] = {}
        for mname in sorted(metric_names):
            metric_first[mname] = {}
            agent_cols = sorted(agent_keys_per_metric.get(mname, set()))
            for ak in agent_cols:
                series: List[float] = []
                for it in iteration_order:
                    series.append(iter_avgs.get(it, {}).get(mname, {}).get(ak))
                metric_first[mname][ak] = series

            # Add default "all_agents" as average across available agent series for each iteration
            if agent_cols:
                all_series: List[float] = []
                filtered_agents = [
                    a for a in agent_cols if a not in ("all", "all_agents")
                ]
                for idx in range(len(iteration_order)):
                    vals: List[float] = []
                    for ak in filtered_agents:
                        v = metric_first[mname].get(ak, [None] * len(iteration_order))[
                            idx
                        ]
                        if v is not None:
                            try:
                                vals.append(float(v))
                            except Exception:
                                pass
                    all_series.append(sum(vals) / len(vals) if vals else None)
                metric_first[mname]["all_agents"] = all_series

        # Write only iteration averages to a single root-level file
        with open(outfile, "w", encoding="utf-8") as f:
            json.dump(metric_first, f, ensure_ascii=False)

    write_iteration_avgs(records)

    return outfile

    # Note: code below is unreachable due to early return, but kept for reference


def _resolve_metric_function(
    spec: str,
    search_modules: List[str],
) -> Tuple[str, Callable[[SimulationStepLog], Optional[Dict[str, float]]]]:
    """
    Resolve a metric specifier to a callable.
    - If spec contains ':' -> treat as module:function
    - Else search provided modules for 'metric_{spec}' or '{spec}' attributes
    Returns (metric_name, callable)
    """
    metric_name = spec
    # Search default modules
    for mod_name in search_modules:
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        if hasattr(mod, metric_name):
            return metric_name, getattr(mod, metric_name)
    raise ValueError(
        f"Could not resolve metric '{spec}'. Provide module:function or ensure it exists in {search_modules}."
    )


def _cli():
    parser = argparse.ArgumentParser(
        description="Stream and compute rollout statistics with functional metrics"
    )
    parser.add_argument(
        "data_root",
        type=str,
        help="Path to experiment/seed folder containing iteration_* subfolders",
    )
    parser.add_argument(
        "metrics",
        nargs="+",
        help="Metric names (e.g., avg_reward avg_greed) or module:function specs",
    )
    # Iteration range controls
    parser.add_argument(
        "--from-iteration",
        type=int,
        default=0,
        help="Start processing from this iteration index (inclusive)",
    )
    parser.add_argument(
        "--last-iteration",
        type=int,
        default=None,
        help="Stop processing at this iteration index (inclusive)",
    )
    parser.add_argument(
        "--search-mod",
        action="append",
        default=[
            "mllm.markov_games.ipd.ipd_statistics",
            "mllm.markov_games.negotiation.negotiation_statistics",
        ],
        help="Additional module to search for metric functions (can be passed multiple times)",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Max parallel processes per iteration (defaults to CPU count)",
    )
    args = parser.parse_args()

    # Build metrics mapping
    metrics: Dict[str, Callable[[SimulationStepLog], Optional[Dict[str, float]]]] = {}
    for spec in args.metrics:
        if not spec:
            continue
        name, fn = _resolve_metric_function(spec, args.search_mod)
        metrics[name] = fn

    out = run_stats_functional(
        Path(args.data_root),
        "custom",
        metrics,
        jobs=args.jobs,
        from_iteration=args.from_iteration,
        last_iteration=args.last_iteration,
    )
    print(str(out))


if __name__ == "__main__":
    _cli()
