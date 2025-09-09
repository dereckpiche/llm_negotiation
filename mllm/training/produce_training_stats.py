import copy
import csv
import gc
import json
import logging
import os
import pickle
import random
import re
import subprocess
import sys
import time
from datetime import datetime
from statistics import mean

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf


def get_from_nested_dict(dictio: dict, path: list[str]):
    for sp in path[:-1]:
        dictio = dictio[sp]
    return dictio.get(path[-1])


def set_at_path(dictio: dict, path: list[str], value):
    for sp in path[:-1]:
        if sp not in dictio:
            dictio[sp] = {}
        dictio = dictio[sp]
    dictio[path[-1]] = value


def produce_tabular_render(inpath: str, outpath: str = None):
    """
    TODO: docstring
    """
    with open(inpath, "r") as f:
        data = json.load(f)
    rollout_paths = data.keys()
    for rollout_path in rollout_paths:
        if outpath is None:
            m_path = rollout_path.replace("/", "|")
            m_path = m_path.replace(".json", "")
            m_path = (
                os.path.split(inpath)[0]
                + "/contextualized_tabular_renders/"
                + m_path
                + "_tabular_render.render.csv"
            )
        # import pdb; pdb.set_trace()
        os.makedirs(os.path.split(m_path)[0], exist_ok=True)
        metrics = data[rollout_path]
        d = {k: [] for k in metrics[0].keys()}
        for m in metrics:
            for k, v in m.items():
                d[k].append(v)
        d = pd.DataFrame(d)
        d.to_csv(m_path)


def get_metric_paths(data: list[dict]):
    d = data[0]
    paths = []

    def traverse_dict(d, current_path=[]):
        for key, value in d.items():
            new_path = current_path + [key]
            if isinstance(value, dict):
                traverse_dict(value, new_path)
            else:
                paths.append(new_path)

    traverse_dict(d)
    return paths


def print_metric_paths(data: list[dict]):
    paths = get_metric_paths(data)
    for p in paths:
        print(p)


def get_metric_iteration_list(data: list[dict], metric_path: list[str]):
    if isinstance(metric_path, str):
        metric_path = [metric_path]
    sgl = []
    for d in data:
        sgl.append(get_from_nested_dict(d, metric_path))
    return sgl


def to_1d_numeric(x):
    """Return a 1-D float array (or None if not numeric). Accepts scalars, numpy arrays, or nested list/tuple of them."""
    if x is None:
        return None
    if isinstance(x, (int, float, np.number)):
        return np.array([float(x)], dtype=float)
    if isinstance(x, np.ndarray):
        try:
            return x.astype(float).ravel()
        except Exception:
            return None
    if isinstance(x, (list, tuple)):
        parts = []
        for e in x:
            arr = to_1d_numeric(e)
            if arr is not None and arr.size > 0:
                parts.append(arr)
        if parts:
            return np.concatenate(parts)
        return None
    return None


def get_single_metric_vector(data, metric_path, iterations=None):
    if isinstance(metric_path, str):
        metric_path = [metric_path]
    if iterations == None:
        iterations = len(data)
    vecs = []
    for d in data:
        ar = get_from_nested_dict(d, metric_path)
        arr = to_1d_numeric(ar)
        if arr is not None:
            vecs.append(arr)

    return np.concatenate(vecs) if vecs else np.empty(0, dtype=float)


def _load_metrics_file(file_path: str):
    if not (file_path.endswith(".tally.pkl") or file_path.endswith(".pkl")):
        raise ValueError("Only *.tally.pkl files are supported.")
    import pickle

    with open(file_path, "rb") as f:
        tree = pickle.load(f)
    return tree


def get_iterations_data(iterations_path: str):
    iterations_data = []
    more_iterations = True
    n = 0
    iteration_path = os.path.join(iterations_path, f"iteration_{n:03d}")
    while more_iterations:
        if os.path.isdir(iteration_path):
            for root, dirs, files in os.walk(iteration_path):
                for file in sorted([f for f in files if f.endswith(".tally.pkl")]):
                    file_path = os.path.join(root, file)
                    iterations_data.append(_load_metrics_file(file_path))
        else:
            more_iterations = False
        n += 1
        iteration_path = os.path.join(iterations_path, f"iteration_{n:03d}")
    return iterations_data


def get_leaf_items(array_tally: dict, prefix: list[str] = None):
    if prefix is None:
        prefix = []
    for key, value in array_tally.items():
        next_prefix = prefix + [str(key)]
        if isinstance(value, dict):
            yield from get_leaf_items(value, next_prefix)
        else:
            yield next_prefix, value


def _sanitize_filename_part(part: str) -> str:
    s = part.replace("/", "|")
    s = s.replace(" ", "_")
    return s


def render_rt_tally_pkl_to_csvs(pkl_path: str, outdir: str):
    with open(pkl_path, "rb") as f:
        payload = pickle.load(f)
    # Backward compatibility: older tallies stored the dict directly
    if isinstance(payload, dict) and "array_tally" in payload:
        array_tally = payload.get("array_tally", {})
        rowmeta = payload.get("rowmeta", {})
        row_ids = payload.get("row_ids", [])
    else:
        array_tally = payload
        rowmeta = {}
        row_ids = []

    os.makedirs(outdir, exist_ok=True)
    trainer_id = os.path.basename(pkl_path).replace(".rt_tally.pkl", "")
    for path_list, rollout_tally_items in get_leaf_items(array_tally):
        # Create file and initiate writer
        path_part = ".".join(_sanitize_filename_part(p) for p in path_list)
        filename = f"{trainer_id}__{path_part}.render.csv"
        out_path = os.path.join(outdir, filename)

        # Write metric rows to CSV
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)

            # Write header row - need to determine metric column count from first rollout_tally_item
            first_item = rollout_tally_items[0]
            metric_cols = (
                first_item.metric_matrix.shape[1]
                if first_item.metric_matrix.ndim > 1
                else 1
            )
            header = ["agent_id", "crn_id", "rollout_id"] + [
                f"t_{i}" for i in range(metric_cols)
            ]
            writer.writerow(header)

            for rollout_tally_item in rollout_tally_items:
                crn_ids = rollout_tally_item.crn_ids
                rollout_ids = rollout_tally_item.rollout_ids
                agent_ids = rollout_tally_item.agent_ids
                metric_matrix = rollout_tally_item.metric_matrix
                for i in range(metric_matrix.shape[0]):
                    row_vals = metric_matrix[i].reshape(-1)
                    # Convert row_vals to a list to avoid numpy concatenation issues
                    row_vals = (
                        row_vals.tolist()
                        if hasattr(row_vals, "tolist")
                        else list(row_vals)
                    )
                    row_prefix = [
                        agent_ids[i],
                        crn_ids[i],
                        rollout_ids[i],
                    ]
                    writer.writerow(row_prefix + row_vals)


def render_tally_pkl_to_csvs(pkl_path: str, outdir: str):
    with open(pkl_path, "rb") as f:
        payload = pickle.load(f)
    # Backward compatibility: older tallies stored the dict directly
    if isinstance(payload, dict) and "array_tally" in payload:
        array_tally = payload.get("array_tally", {})
        rowmeta = payload.get("rowmeta", {})
        row_ids = payload.get("row_ids", [])
    else:
        array_tally = payload
        rowmeta = {}
        row_ids = []
    os.makedirs(outdir, exist_ok=True)
    trainer_id = os.path.basename(pkl_path).replace(".tally.pkl", "")
    for path_list, array_list in get_leaf_items(array_tally):
        # Build datapoints by expanding element-wise: vectors → 1 row, matrices → per-row
        datapoints = []
        for item in array_list:
            # Normalize item
            if isinstance(item, (int, float)):
                arr = np.asarray([item])
            else:
                arr = np.asarray(item)
            # Numeric arrays
            if arr.ndim == 0:
                datapoints.append(arr.reshape(1))
            elif arr.ndim == 1:
                datapoints.append(arr)
            else:
                for i in range(arr.shape[0]):
                    datapoints.append(arr[i].reshape(-1))
        # Build filename
        path_part = ".".join(_sanitize_filename_part(p) for p in path_list)
        filename = f"{trainer_id}__{path_part}.render.csv"
        out_path = os.path.join(outdir, filename)
        # Write CSV
        with open(out_path, "w", newline="") as f:
            import csv

            writer = csv.writer(f)
            # Determine max length after expansion for header
            max_len = 0
            for r in datapoints:
                max_len = max(max_len, int(np.asarray(r).size))
            for i, r in enumerate(datapoints):
                r_arr = np.asarray(r)
                if r_arr.size < max_len:
                    pad = np.empty((max_len - r_arr.size,), dtype=r_arr.dtype)
                    if pad.dtype == object:
                        pad[:] = ""
                    else:
                        pad[:] = np.nan
                    r_arr = np.concatenate([r_arr, pad])
                row_vals = [
                    x if not isinstance(x, (np.floating, np.integer)) else x.item()
                    for x in r_arr
                ]
                writer.writerow(row_vals)


def render_iteration_trainer_stats(iteration_dir: str, outdir: str | None = None):
    input_dir = iteration_dir
    output_dir = outdir or os.path.join(iteration_dir, "trainer_stats.render.")
    os.makedirs(output_dir, exist_ok=True)
    for fname in sorted(os.listdir(input_dir)):
        if fname.endswith(".rt_tally.pkl"):
            pkl_path = os.path.join(input_dir, fname)
            render_rt_tally_pkl_to_csvs(pkl_path=pkl_path, outdir=output_dir)
        elif fname.endswith(".tally.pkl"):
            pkl_path = os.path.join(input_dir, fname)
            render_tally_pkl_to_csvs(pkl_path=pkl_path, outdir=output_dir)
