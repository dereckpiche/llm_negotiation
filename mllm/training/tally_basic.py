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
        self.base_tally = {}

    def reset(self):
        """
        Resets the base and contextualized tallies to empty dictionaries.
        """
        self.base_tally = {}

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

    def add_metric(self, path: str, metric: Union[float, int, str, np.ndarray, dict]):
        """
        Adds a metric to the base tally at the specified path.

        Args:
            path (list): List of keys representing the path in the base tally.
            metric (float|int|str|np.ndarray|dict): The metric value to add.
        """
        metric = deepcopy(metric)
        assert isinstance(
            metric, Union[float, int, str, np.ndarray, dict]
        ), "Metric of incorrect type"

        current_metric = self.get_from_nested_dict(dictio=self.base_tally, path=path)

        if isinstance(metric, np.ndarray):
            metric = metric.tolist()

        if current_metric == None:
            self.set_at_path(dictio=self.base_tally, path=path, value=[metric])
        else:
            current_metric.append(metric)

    def save(self, path: str):
        """
        Saves the base and contextualized tallies to disk as JSON files, and also saves contextualized tallies as CSV files for each game/rollout.

        Args:
            path (str): Directory path where the metrics will be saved.
        """
        os.makedirs(name=path, exist_ok=True)

        from datetime import datetime

        now = datetime.now()

        savepath = os.path.join(
            path, f"basic_training_metrics_{now:%Y-%m-%d___%H-%M-%S}.json"
        )
        with open(savepath, "w") as fp:
            json.dump(self.base_tally, fp, indent=4)
