import json
import os
from copy import deepcopy
from typing import Union

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer


class RolloutTallyItem:
    def __init__(self, crn_ids: list[str], rollout_ids: list[str], agent_ids: list[str], metric_matrix: torch.Tensor): 
        """
        Initializes the RolloutTallyItem object.

        Args:
            crn_ids (list[str]): List of CRN IDs.
            rollout_ids (list[str]): List of rollout IDs.
            agent_ids (list[str]): List of agent IDs.
            metric_matrix (torch.Tensor): Metric matrix.
        """
        if isinstance(crn_ids, torch.Tensor):
            crn_ids = crn_ids.detach().cpu().numpy()
        if isinstance(rollout_ids, torch.Tensor):
            rollout_ids = rollout_ids.detach().cpu().numpy()
        if isinstance(agent_ids, torch.Tensor):
            agent_ids = agent_ids.detach().cpu().numpy()
        self.crn_ids = crn_ids
        self.rollout_ids = rollout_ids
        self.agent_ids = agent_ids
        metric_matrix = metric_matrix.detach().cpu()
        assert 0 < metric_matrix.ndim <= 2, "Metric matrix must have less than or equal to 2 dimensions"
        if metric_matrix.ndim == 1:
            metric_matrix = metric_matrix.reshape(1, -1)
        # Convert to float32 if tensor is in BFloat16 format (not supported by numpy)
        if metric_matrix.dtype == torch.bfloat16:
            metric_matrix = metric_matrix.float()
        self.metric_matrix = metric_matrix.numpy()

class RolloutTally:
    """
    Tally is a utility class for collecting and storing training metrics.
    It supports adding metrics at specified paths and saving them to disk.
    """

    def __init__(self):
        """
        Initializes the RolloutTally object.

        Args:
            tokenizer (AutoTokenizer): Tokenizer for converting token IDs to strings.
            max_context_length (int, optional): Maximum context length for contextualized metrics. Defaults to 30.
        """
        # Array-preserving structure (leaf lists hold numpy arrays / scalars)
        self.metrics = {}
        # Global ordered list of sample identifiers (crn_id, rollout_id) added in the order samples are processed

    def reset(self):
        """
        Resets the base and contextualized tallies to empty dictionaries.
        """
        self.metrics = {}

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
        self, path: list[str], rollout_tally_item: RolloutTallyItem
    ):
        """
        Adds a metric to the base tally at the specified path.

        Args:
            path (list): List of keys representing the path in the base tally.
            rollout_tally_item (RolloutTallyItem): The rollout tally item to add.
        """
        rollout_tally_item = deepcopy(rollout_tally_item)

        # Update array-preserving tally
        array_list = self.get_from_nested_dict(dictio=self.metrics, path=path)
        if array_list is None:
            self.set_at_path(dictio=self.metrics, path=path, value=[rollout_tally_item])
        else:
            array_list.append(rollout_tally_item)


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

            pkl_path = os.path.join(folder, f"{identifier}.rt_tally.pkl")
            payload = {"metrics": self.metrics}
            with open(pkl_path, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            pass
