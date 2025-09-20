from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import pickle
import shutil
import sys
import textwrap
import urllib.error
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

try:
    import numpy as np
except ImportError:
    np = None

from mllm.markov_games.rollout_tree import RolloutTreeRootNode

# Optional progress bar
try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    tqdm = None


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


def find_iteration_folders(
    global_folder: str | Path,
    from_iteration: int = 0,
    last_iteration: Optional[int] = None,
) -> List[Path]:
    """Find all iteration_* folders within the folder or under seed_*.

    Filters by numeric iteration index (parsed from folder suffix) using
    from_iteration (inclusive) and last_iteration (inclusive) if provided.
    Works whether global_folder is a seed directory or a parent containing seeds.
    """
    base = Path(global_folder)

    candidates: List[Path] = []

    # Search in the folder itself
    for item in base.glob("iteration_*"):
        if item.is_dir():
            candidates.append(item)

    # Search in seed_* subdirectories
    for seed_dir in base.glob("seed_*/"):
        if seed_dir.is_dir():
            for item in seed_dir.glob("iteration_*"):
                if item.is_dir():
                    candidates.append(item)

    # Parse numeric iteration indices and filter
    def parse_idx(p: Path) -> Optional[int]:
        name = p.name
        try:
            return int(name.split("_")[-1])
        except Exception:
            return None

    filtered: List[tuple[int, Path]] = []
    for p in candidates:
        idx = parse_idx(p)
        if idx is None:
            continue
        if idx < from_iteration:
            continue
        if (last_iteration is not None) and (idx > last_iteration):
            continue
        filtered.append((idx, p))

    # Sort numerically by idx
    filtered.sort(key=lambda t: t[0])
    return [p for (_idx, p) in filtered]


def discover_metric_functions(module_name: str) -> Dict[str, Callable[[Any], Any]]:
    """Import a statistics module and return all public callables as metrics.

    Assumes every method in the stats module is a metric function.
    """
    mod = importlib.import_module(module_name)
    metrics: Dict[str, Callable[[Any], Any]] = {}
    # Prefer explicit exported list if present
    export_list = getattr(mod, "stat_functs", None)
    if isinstance(export_list, (list, tuple)) and export_list:
        for name in export_list:
            try:
                attr = getattr(mod, name)
            except Exception:
                continue
            if callable(attr):
                metrics[name] = attr
        return metrics
    # Fallback: discover all public callables (legacy behavior)
    for attr_name in dir(mod):
        if attr_name.startswith("_"):
            continue
        try:
            attr = getattr(mod, attr_name)
        except Exception:
            continue
        if callable(attr):
            metrics[attr_name] = attr
    return metrics


# -----------------------------
# Inlined stats engine (from stats.py)
# -----------------------------


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


def stream_rollout_files(iteration_folder: Path) -> Iterator[Path]:
    for p in iteration_folder.rglob("*.rt.pkl"):
        if p.is_file():
            yield p


def load_root(path: Path) -> dict:
    """Load rollout tree as raw dict without Pydantic validation (fast)."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def _compute_record_for_file(
    pkl_path: str,
    iteration_name: str,
    metrics_items: List[Tuple[str, Callable[[Any], Optional[Dict[str, float]]]]],
):
    root = load_root(Path(pkl_path))
    # Ultra-minimal aggregation: call metric, stash raw (coerced) values per agent.
    agg: Dict[str, Dict[str, List[Any]]] = {m: {} for m, _ in metrics_items}
    for sl in iterate_main_simulation_logs_fast(root):
        for mname, fn in metrics_items:
            try:
                vals = fn(sl)
            except Exception:
                continue
            if isinstance(vals, dict):
                for aid, v in vals.items():
                    agg[mname].setdefault(str(aid), []).append(v)

    # finalize averages per metric/agent for this rollout
    result: Dict[str, Dict[str, Optional[float]]] = {}
    for mname, _ in metrics_items:
        result[mname] = {}
        for aid, vals in agg[mname].items():
            nums = [
                float(v)
                for v in vals
                if isinstance(v, (int, float))
                and not isinstance(v, bool)
                and v is not None
                and not (isinstance(v, float) and math.isnan(v))
            ]
            result[mname][aid] = (sum(nums) / len(nums)) if nums else None

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
    metrics: Dict[str, Callable[[Any], Optional[Dict[str, float]]]],
    jobs: Optional[int] = None,
    from_iteration: int = 0,
    last_iteration: Optional[int] = None,
) -> Path:
    data_root = Path(data_root)
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

    records: List[Dict[str, Any]] = []

    # Prepare tasks across all iterations to use a single process pool
    metrics_items = list(metrics.items())
    max_workers = jobs or max(1, (os.cpu_count() or 2))
    tasks: List[Tuple[str, str]] = []  # (pkl_path, iteration_name)
    for iteration_folder in iteration_folders:
        iteration_name = Path(iteration_folder).name
        files = sorted(
            (p for p in stream_rollout_files(Path(iteration_folder))),
            key=lambda x: str(x),
        )
        for p in files:
            tasks.append((str(p), iteration_name))

    # Sort tasks deterministically by (iteration_name, path)
    tasks.sort(key=lambda t: (t[1], t[0]))

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

    # Write per-iteration averages (across mgids) as JSON
    def write_iteration_avgs(records_list: List[Dict[str, Any]]):
        by_iter: Dict[str, List[Dict[str, Any]]] = {}
        iteration_order: List[str] = []
        for r in records_list:
            it = str(r.get("iteration"))
            if it not in by_iter:
                by_iter[it] = []
                iteration_order.append(it)
            by_iter[it].append(r)

        iteration_order = sorted(iteration_order)

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
                        if not isinstance(stats, dict):
                            continue
                        if ak not in stats:
                            continue
                        val = stats[ak]
                        if val is None or isinstance(val, bool):
                            continue
                        try:
                            fval = float(val)
                        except Exception:
                            continue
                        if math.isnan(fval):
                            continue
                        vals.append(fval)
                    iter_avgs[it_name][mname][ak] = (
                        (sum(vals) / len(vals)) if vals else None
                    )

        metric_first: Dict[str, Dict[str, List[float]]] = {}
        for mname in sorted(metric_names):
            metric_first[mname] = {}
            agent_cols = sorted(agent_keys_per_metric.get(mname, set()))
            for ak in agent_cols:
                series: List[float] = []
                for it in iteration_order:
                    series.append(iter_avgs.get(it, {}).get(mname, {}).get(ak))
                metric_first[mname][ak] = series

        with open(outfile, "w", encoding="utf-8") as f:
            json.dump(metric_first, f, ensure_ascii=False, indent=2)

    write_iteration_avgs(records)

    return outfile


def compute_stats_and_stage(
    data_root: Path,
    game_kind: str,
    metrics: Dict[str, Callable[[Any], Any]],
    jobs: Optional[int],
    from_iteration: int,
    last_iteration: Optional[int],
) -> Path:
    """Compute statistics.json and stage into 0A_paperdata_for_<experiment>."""
    out_path = run_stats_functional(
        data_root,
        game_kind,
        metrics,
        jobs=jobs,
        from_iteration=from_iteration,
        last_iteration=last_iteration,
    )
    experiment_name = (
        data_root.name
        if not data_root.name.startswith("seed_")
        else data_root.parent.name
    )
    stats_dir = data_root / f"0A_paperdata_for_{_sanitize_filename(experiment_name)}"
    stats_dir.mkdir(parents=True, exist_ok=True)
    staged = stats_dir / "statistics.json"
    try:
        if staged.exists():
            staged.unlink()
    except Exception:
        pass
    shutil.copy2(out_path, staged)
    return out_path


def _sanitize_filename(name: str) -> str:
    safe = "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in name)
    return safe or "metric"


STYLE_URL = "https://raw.githubusercontent.com/dereckpiche/DedeStyle/refs/heads/main/dedestyle.mplstyle"


def plot_statistics_json(seed_root: Path) -> None:
    """Generate plots from statistics.json into seed_root/0B_plots.render.

    Expects statistics.json structure: {metric: {agent: [series...]}}.
    Looks for statistics.json inside 0A_paperdata_for_<experiment> first.
    """
    experiment_name = (
        seed_root.name
        if not seed_root.name.startswith("seed_")
        else seed_root.parent.name
    )
    stats_dir = seed_root / f"0A_paperdata_for_{_sanitize_filename(experiment_name)}"
    stats_json_path = stats_dir / "statistics.json"
    if not stats_json_path.exists():
        legacy = seed_root / "statistics.json"
        if legacy.exists():
            stats_json_path = legacy
            stats_dir = seed_root
        else:
            print(f"No statistics.json found (checked {stats_dir} and root)")
            return

    try:
        # Use non-interactive backend for headless render
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
        import matplotlib.style as mplstyle  # type: ignore
    except Exception as e:
        print(f"Matplotlib not available for plots: {e}")
        return

    with open(stats_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    render_dir = seed_root / "0B_plots"
    render_dir.mkdir(parents=True, exist_ok=True)
    # Numpy tensor files now stored directly in stats_dir (no subdirectory)
    tensors_dir = stats_dir
    # Copy provenance folders if present into paperdata_origins
    origins_dir = stats_dir / "paperdata_origins"
    origins_dir.mkdir(parents=True, exist_ok=True)
    for folder_name in [".hydra", "src_code_for_reproducibility"]:
        # Prefer folder inside seed_root; if missing, fallback to parent of seed_root
        candidates = [seed_root / folder_name, seed_root.parent / folder_name]
        src_path = next((p for p in candidates if p.exists() and p.is_dir()), None)
        if src_path is None:
            continue
        dest_path = origins_dir / folder_name
        if dest_path.exists():
            # Skip if already copied (assume unchanged)
            continue
        try:
            shutil.copytree(src_path, dest_path)
        except FileExistsError:
            pass
        except Exception as e:
            print(f"Failed to copy {folder_name} provenance from {src_path}: {e}")
    # Create experiment description placeholder if not already
    desc_file = stats_dir / "experiment_description.txt"
    if not desc_file.exists():
        try:
            desc_file.write_text(
                "Description of the experiment here..\n", encoding="utf-8"
            )
        except Exception as e:
            print(f"Failed to write experiment description: {e}")

    # Attempt to download and apply the custom style
    style_path = stats_dir / "dedestyle.mplstyle"
    try:
        # Download only if missing or empty
        if (not style_path.exists()) or (style_path.stat().st_size == 0):
            with urllib.request.urlopen(STYLE_URL, timeout=10) as resp:
                content = resp.read()
            with open(style_path, "wb") as sf:
                sf.write(content)
        mplstyle.use(str(style_path))
    except Exception as e:
        print(f"Could not apply style from {STYLE_URL}: {e}")

    # For each metric, plot each agent's series
    for metric_name, agent_to_series in sorted((data or {}).items()):
        if not isinstance(agent_to_series, dict):
            continue
        # Determine which agents to plot: if all_agents present alongside others, drop it.
        agent_keys = sorted(agent_to_series.keys())
        if "all_agents" in agent_keys and len(agent_keys) > 1:
            agent_keys = [k for k in agent_keys if k != "all_agents"]
        # Build x axis by series length (assume all agents same length; if not, use max)
        max_len = 0
        for k, series in agent_to_series.items():
            if k not in agent_keys:
                continue
            try:
                max_len = max(max_len, len(series) if series is not None else 0)
            except Exception:
                continue
        if max_len == 0:
            continue
        x = list(range(max_len))

        fig, ax = plt.subplots(figsize=(8, 4.5))
        for agent_name in agent_keys:
            series = agent_to_series.get(agent_name)
            if not isinstance(series, list) or len(series) == 0:
                continue
            # Pad/truncate to x length; matplotlib treats None as gaps
            y = list(series[:max_len])
            if len(y) < max_len:
                y = y + [None] * (max_len - len(y))
            try:
                ax.plot(x, y, label=str(agent_name), linewidth=1.0, markersize=2.0)
            except Exception:
                # Best-effort plotting
                continue

        ax.set_xlabel("gradient_steps")
        # Force integer ticks for discrete gradient steps
        try:
            from matplotlib.ticker import MaxNLocator  # type: ignore

            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        except Exception:
            pass
        # Show metric name as title instead of y-axis label
        ax.set_title(str(metric_name))
        ax.set_ylabel("")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        out_file = render_dir / f"{_sanitize_filename(metric_name)}.png"
        try:
            fig.savefig(out_file, dpi=150)
        finally:
            plt.close(fig)

        # Export per-agent numpy tensors for this metric (one file per agent)
        if np is not None:
            try:
                for agent_name in sorted(agent_to_series.keys()):
                    series = agent_to_series.get(agent_name)
                    if not isinstance(series, list) or len(series) == 0:
                        continue
                    clean_series = []
                    for val in series:
                        if val is None or (isinstance(val, float) and math.isnan(val)):
                            clean_series.append(np.nan)
                        else:
                            try:
                                clean_series.append(float(val))
                            except (ValueError, TypeError):
                                clean_series.append(np.nan)
                    if not clean_series:
                        continue
                    tensor_file = (
                        tensors_dir
                        / f"{_sanitize_filename(metric_name)}_{_sanitize_filename(str(agent_name))}.npy"
                    )
                    np.save(tensor_file, np.array(clean_series))
            except Exception as e:
                print(f"Failed to export tensors for metric {metric_name}: {e}")


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
        description="Compute stats, generate plots, and render artifacts for Markov games",
    )
    parser.add_argument(
        "path",
        nargs="?",
        help="Experiment root or seed folder containing iteration_* (default: .)",
    )
    # Game selection
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--ipd", action="store_true", help="Use IPD statistics module")
    g.add_argument(
        "--nego", action="store_true", help="Use Negotiation statistics module"
    )

    # Optional controls
    parser.add_argument(
        "--jobs", type=int, default=None, help="Parallel workers for stats"
    )
    parser.add_argument(
        "--from-iteration",
        type=int,
        default=0,
        help="Start at iteration index (inclusive)",
    )
    parser.add_argument(
        "--last-iteration",
        type=int,
        default=None,
        help="Stop at iteration index (inclusive)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete all '.render.' artifacts under path and exit",
    )

    # Rendering options
    parser.add_argument(
        "--per-agent",
        action="store_true",
        default=True,
        help="Write per-agent transcripts",
    )
    parser.add_argument(
        "--no-per-agent",
        action="store_false",
        dest="per_agent",
        help="Disable per-agent transcripts",
    )
    parser.add_argument(
        "--include-state-end",
        action="store_true",
        default=False,
        help="Annotate <STATE_END> on lines",
    )
    parser.add_argument(
        "--sim-csv",
        action="store_true",
        default=True,
        help="Export simulation infos to CSV",
    )
    parser.add_argument(
        "--no-sim-csv",
        action="store_false",
        dest="sim_csv",
        help="Do not export simulation infos to CSV",
    )

    args = parser.parse_args()

    data_root = Path(args.path or ".").resolve()

    if args.clean:
        print(f"Cleaning under: {data_root}")
        removed = clean_render_artifacts(data_root)
        print(f"Cleaning complete. Removed {removed} item(s) containing '.render.'")
        return

    # 1) Discover metrics based on game kind
    if args.ipd:
        stats_mod = "mllm.markov_games.ipd.ipd_statistics"
        game_kind = "ipd"
    else:
        stats_mod = "mllm.markov_games.negotiation.negotiation_statistics"
        game_kind = "negotiation"

    metrics = discover_metric_functions(stats_mod)
    if not metrics:
        print(f"No metrics discovered in {stats_mod}. Aborting.")
        return

    # Determine seed folders: if path itself looks like a seed (contains iteration_*), use it; else use all seed_*
    seeds: List[Path] = []
    has_iterations_here = any((data_root.glob("iteration_*")))
    if has_iterations_here:
        seeds = [data_root]
    else:
        seeds = sorted(
            [p for p in data_root.glob("seed_*/") if p.is_dir()], key=lambda p: str(p)
        )
        if not seeds:
            print(
                f"No seed folders found under {data_root} and no iterations at root. Nothing to do."
            )
            return

    for seed_root in seeds:
        print(f"Computing statistics.json for seed: {seed_root}")
        stats_out = compute_stats_and_stage(
            data_root=seed_root,
            game_kind=game_kind,
            metrics=metrics,
            jobs=args.jobs,
            from_iteration=args.from_iteration,
            last_iteration=args.last_iteration,
        )
        print(f"Wrote: {stats_out}")
        print(
            "Generating plots in 0B_plots and paper data in 0A_paperdata_for_<exp> ..."
        )
        plot_statistics_json(seed_root)

        iteration_folders = find_iteration_folders(
            seed_root,
            from_iteration=args.from_iteration,
            last_iteration=args.last_iteration,
        )
        if not iteration_folders:
            print(f"No iteration_* folders found under {seed_root}")
            continue
        print(f"Found {len(iteration_folders)} iteration folders under {seed_root}")

        successful_count = 0
        for folder in iteration_folders:
            if process_single_folder(
                folder,
                None,
                per_agent=args.per_agent,
                include_state_end=args.include_state_end,
                sim_csv=args.sim_csv,
                recursive=False,
            ):
                successful_count += 1

        print(
            f"Seed {seed_root.name}: processed {successful_count}/{len(iteration_folders)} iteration folders."
        )


if __name__ == "__main__":
    main()
