import json
import os
from copy import deepcopy
from typing import Union

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer


class Tally:
    """
    Tally is a utility class for collecting and storing training metrics.
    It supports adding metrics at specified paths and saving them to disk.
    """

    def __init__(self):
        """
        Initializes the Tally object.

        Args:
            tokenizer (AutoTokenizer): Tokenizer for converting token IDs to strings.
            max_context_length (int, optional): Maximum context length for contextualized metrics. Defaults to 30.
        """
        # Array-preserving structure (leaf lists hold numpy arrays / scalars)
        self.array_tally = {}

    def reset(self):
        """
        Resets the base and contextualized tallies to empty dictionaries.
        """
        self.array_tally = {}

    def get_from_nested_dict(self, dictio: dict, path: str):
        """
        Retrieves the value at a nested path in a dictionary.

        Args:
            dictio (dict): The dictionary to search.
            path (list): List of keys representing the path.

        Returns:
            Any: The value at the specified path, or None if not found.
        """
        assert isinstance(path, list), "Path must be list."
        for sp in path[:-1]:
            dictio = dictio.setdefault(sp, {})
        return dictio.get(path[-1], None)

    def set_at_path(self, dictio: dict, path: str, value):
        """
        Sets a value at a nested path in a dictionary, creating intermediate dictionaries as needed.

        Args:
            dictio (dict): The dictionary to modify.
            path (list): List of keys representing the path.
            value (Any): The value to set at the specified path.
        """
        for sp in path[:-1]:
            dictio = dictio.setdefault(sp, {})
        dictio[path[-1]] = value

    def add_metric(
        self, path: str, metric: Union[float, int, np.ndarray, torch.Tensor, list]
    ):
        """
        Adds a metric to the base tally at the specified path.

        Args:
            path (list): List of keys representing the path in the base tally.
            metric (float|int|str|np.ndarray|dict): The metric value to add.
        """
        metric = deepcopy(metric)

        # Array-only: accept numbers, tensors, numpy arrays, lists (will convert). No strings.
        allowed_types = (float, int, np.ndarray, torch.Tensor, list)
        assert isinstance(metric, allowed_types), "Metric of incorrect type"

        # Prepare array-preserving representation only
        array_metric = metric

        if isinstance(metric, torch.Tensor):
            if metric.dim() == 0:
                array_metric = np.asarray(metric.item())
            else:
                array_metric = metric.detach().cpu().numpy()

        if isinstance(array_metric, (float, int, np.number)):
            array_metric = np.asarray(array_metric)
        elif isinstance(array_metric, list):
            # convert lists to numpy arrays; may be object dtype for ragged
            try:
                array_metric = np.asarray(array_metric)
            except Exception:
                array_metric = np.array(array_metric, dtype=object)

        # Update array-preserving tally
        array_list = self.get_from_nested_dict(dictio=self.array_tally, path=path)
        if array_list is None:
            self.set_at_path(dictio=self.array_tally, path=path, value=[array_metric])
        else:
            array_list.append(array_metric)

    def save(self, identifier: str, folder: str):
        """
        Saves the base and contextualized tallies to disk as JSON files, and also saves contextualized tallies as CSV files for each game/rollout.

        Args:
            path (str): Directory path where the metrics will be saved.
        """
        os.makedirs(name=folder, exist_ok=True)

        from datetime import datetime

        now = datetime.now()

        # Pickle only (fastest, exact structure with numpy/scalars at leaves)
        try:
            import pickle

            pkl_path = os.path.join(folder, f"{identifier}.tally.pkl")
            with open(pkl_path, "wb") as f:
                pickle.dump(self.array_tally, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            pass
