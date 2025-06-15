import copy
import json
import os
from typing import Union
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer


class RtTally:
    def __init__(
        self, 
        tokenizer: AutoTokenizer, 
        max_context_length: int = 30):
        self.tokenizer = tokenizer
        self.max_context_length = max_context_length
        self.base_tally = {}
        self.contextualized_tally = {}


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

    def add_metric(
        self, 
        path: str, 
        metric: Union[float, int, str, np.ndarray, dict]):
        """
        TODO: docstring
        """
        assert isinstance(
            metric, Union[float, int, str, np.ndarray, dict]
        ), "Metric of incorrect type"

        current_metric = self.get_at_path(dictio=self.base_tally, path=path)

        if isinstance(metric, np.ndarray): 
            metric = metric.tolist()

        if current_metric == None:
            self.set_at_path(dictio=self.base_tally, path=path, value=[metric])
        else:
            current_metric.append(metric)
            
    def add_contextualized_token_metrics(
        self,
        game_ids: list[str],
        metric_id: str,
        contexts: torch.Tensor,
        metrics: torch.Tensor,
        action_mask: torch.Tensor,
        to_tids: bool= False
    ):
        """
        TODO: docstring
        """

        # if len(contexts.shape) == 1:
        #     contexts = contexts.unsqueeze(0)
        # if len(metrics.shape) == 1:
        #     metrics = metrics.unsqueeze(0)

        # assert len(contexts.shape) == 2, "Contexts tensor does not have the right shape"
        # assert len(metrics.shape) == 2, "Metrics tensor does not have the right shape"

        if len(metrics.shape) == 2:
            B, S = metrics.shape
        else:
            B, S, _ = metrics.shape

        counter = 0
        for i in range(B):
            rollout_id =  game_ids[i]
            self.contextualized_tally.setdefault(rollout_id, [])
            rollout_data = self.contextualized_tally.get(rollout_id)
            for j in range(S):
                if action_mask[i, j].item() != 0:

                    ctx = contexts[i, j+1 - min(j, self.max_context_length) : j+1].squeeze()
                    context_string = self.tids_to_str(ctx.tolist())
                    value = metrics[i, j]
                    if isinstance(value, Union[np.ndarray, torch.Tensor]): 
                        value = value.tolist()
                    if to_tids:
                        value = self.tids_to_str(value)
                    # TODO: catch context overflows
                    context = context_string[:-1]
                    next_token = context_string[-1]

                    if len(rollout_data) <= counter:
                        dictio = {
                            "context": context,
                            "next_token": next_token,
                            metric_id: value,
                        }
                        rollout_data.append(dictio)
                    else:
                        dictio = rollout_data[counter]
                        assert dictio["context"] == context
                        dictio[metric_id] = value
                    counter += 1

    def save(self, path: str):
        # os.makedirs(name=path, exist_ok=True)
        os.makedirs(
            name=path, 
            exist_ok=True)

        from datetime import datetime
        now = datetime.now()

        savepath = os.path.join(
            path, 
            f"basic_training_metrics_{now:%Y-%m-%d___%H-%M-%S}.json"
        )
        with open(savepath, "w") as fp:
            json.dump(self.base_tally, fp, indent=4)

        savepath = os.path.join(
            path, 
            f"contextualized_training_metrics_{now:%Y-%m-%d___%H-%M-%S}.json"
        )
        with open(savepath, "w") as fp:
            json.dump(self.contextualized_tally, fp, indent=4)

        data = self.contextualized_tally
        game_ids = data.keys()
        for g_id in game_ids:
            m_path = (
                os.path.split(savepath)[0]
                + "/contextualized_tabular_renders/"
                + str(g_id)
                + "_tabular_render.csv"
            )
            # import pdb; pdb.set_trace()
            os.makedirs(os.path.split(m_path)[0], exist_ok=True)
            metrics = data[g_id]
            d = {k: [] for k in metrics[0].keys()}
            for m in metrics:
                for k, v in m.items():
                    d[k].append(v)
            d = pd.DataFrame(d)
            d.to_csv(m_path)
