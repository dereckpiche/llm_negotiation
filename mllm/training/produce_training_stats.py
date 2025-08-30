import copy
import gc
import json
import logging
import os
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
                + "_tabular_render.csv"
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
