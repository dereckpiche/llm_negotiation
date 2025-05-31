import copy
import json
import os
from typing import Union

import numpy as np
import torch
from transformers import AutoTokenizer


class RtTally:
    def __init__(self, tokenizer: AutoTokenizer, max_context_length: int = 10):
        self.tokenizer = tokenizer
        self.max_context_length = max_context_length
        self.tally = {}

    # def set_top_k(self, logits: torch.Tensor):
    #     """
    #     TODO: docstring
    #     """
    #     self.top_k_tids = torch.argmax(logits, dim=-1).squeeze()
    #     B, S = self.top_k_ids.shape
    #     top_k_tids = [tensor.tolist(self.top_k_tids[i].squeeze()) for i in range(B)]
    #     self.top_k_tokens = self.tokenizer.batch_decode(top_k_tids)
    #     import pdb

    #     pdb.set_trace()

    def tids_to_str(self, tids: list[int]):
        """
        TODO: docstring
        """
        token_str = self.tokenizer.convert_ids_to_tokens(tids)
        return token_str

    def get_at_path(self, dictio: dict, path: str):
        """
        TODO: docstring
        """
        assert isinstance(path, list), "Path must be list."
        for sp in path[:-1]:
            dictio = dictio.setdefault(sp, {})
        return dictio.get(path[-1], None)

    def set_at_path(self, dictio: dict, path: str, value):
        for sp in path[:-1]:
            dictio = dictio.setdefault(sp, {})
        dictio[path[-1]] = value

    def add_metric(self, path: str, metric: Union[float, int, str, np.ndarray, dict]):
        """
        TODO: docstring
        """
        assert isinstance(
            metric, Union[float, int, str, np.ndarray, dict]
        ), "Metric of incorrect type"
        current_metric = self.get_at_path(dictio=self.tally, path=path)
        if isinstance(metric, Union[np.ndarray, torch.Tensor]):
            metric = list(metric)
        elif current_metric == None:
            self.set_at_path(dictio=self.tally, path=path, value=metric)
        elif isinstance(current_metric, list) and not isinstance(metric, list):
            current_metric.append(metric)
            self.set_at_path(dictio=self.tally, path=path, value=current_metric)
        else:
            self.set_at_path(
                dictio=self.tally, path=path, value=[current_metric, metric]
            )

    def add_contextualized_token_metrics(
        self,
        data_id: str,
        metric_id: str,
        contexts: torch.Tensor,
        metrics: torch.Tensor,
        action_mask: torch.Tensor,
    ):
        """
        TODO: docstring
        """

        if len(contexts.shape) == 1:
            contexts = contexts.unsqueeze(0)
        if len(metrics.shape) == 1:
            metrics = metrics.unsqueeze(0)

        assert len(contexts.shape) == 2, "Contexts tensor does not have the right shape"
        assert len(metrics.shape) == 2, "Metrics tensor does not have the right shape"

        B, S = metrics.shape

        present == (self.get_at_path(path=path) is not None)

        if not present:
            self.set_at_path(dictio=self.tally, path=path, value=[])

        counter = 0
        for i in range(B):
            for j in range(S):
                if action_mask[i, j].item() != 0:
                    ctx = contexts[i, j - min(j, self.max_context_length) : j].squeeze()
                    context_string = self.tids_to_str(ctx.tolist())
                    value = metrics[i, j].item()
                    # TODO: catch context overflows
                    context = context_string[:-1]
                    next_token = context_string[-1]

                    if not present:
                        # Initialize the dict
                        metric = {
                            "context": context,
                            "next_token": next_token,
                            metric_id: value,
                        }
                        self.add_metric(path=path, metric=metric)

                    else:
                        # Dictionary already present, add metric for that context
                        ar = self.get_at_path(dictio=self.tally, path=path, value=[])
                        l = len(ar)
                        dictio = ar[counter]
                        assert dictio["context"] == context
                        dictio[metric_id] = value
                    counter += 1

    def save(self, path: str):
        # os.makedirs(name=path, exist_ok=True)
        with open(path, "w") as fp:
            json.dump(self.tally, fp)
