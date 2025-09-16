import copy
from dataclasses import dataclass
from typing import Any, Dict, List

from numpy.random import default_rng

from mllm.markov_games.negotiation.nego_simulation import (
    NegotiationObs,
    NegotiationSimulation,
    NegotiationState,
    Split,
    compute_tas_style_rewards,
)

AgentId = str


@dataclass
class TrustAndSplitState(NegotiationState):
    pass


@dataclass
class TrustAndSplitObs(NegotiationObs):
    pass


class TrustAndSplitSimulation(NegotiationSimulation):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def _sample_values(self) -> Dict[AgentId, float]:
        # Independent random per-coin values between 1 and 20 (inclusive)
        return {aid: float(int(self.rng.integers(1, 21))) for aid in self.agent_ids}

    def set_new_round_of_variant(self):
        self.state.values = self._sample_values()
        self.state.quantities = {"coins": 10}
        self.state.split_phase = False

    def get_info_of_variant(
        self, state: NegotiationState, actions: Dict[AgentId, Any]
    ) -> Dict[str, Any]:
        return {
            "quantities": copy.deepcopy(state.quantities),
            "values": copy.deepcopy(state.values),
            "previous_values": copy.deepcopy(state.previous_values),
            "splits": copy.deepcopy(state.splits),
        }

    def get_rewards(self, splits: Dict[AgentId, Split]) -> Dict[AgentId, float]:
        return compute_tas_style_rewards(
            self.agent_ids, self.state.values, splits, self.state.quantities
        )

    def get_obs(self):
        return {agent_id: self.get_obs_agent(agent_id) for agent_id in self.agent_ids}

    def get_obs_agent(self, agent_id):
        other_id = self._other(agent_id)
        last_value_coagent = (
            None
            if self.state.previous_values is None
            else self.state.previous_values.get(other_id)
        )
        last_points_coagent = (
            None
            if self.state.previous_points is None
            else round(self.state.previous_points.get(other_id), 1)
        )
        last_value_agent = (
            None
            if self.state.previous_values is None
            else self.state.previous_values.get(agent_id)
        )
        last_points_agent = (
            None
            if self.state.previous_points is None
            else round(self.state.previous_points.get(agent_id), 1)
        )
        last_split_coagent = None
        last_split_agent = None
        if self.state.previous_splits is not None:
            last_split_coagent = self.state.previous_splits[
                other_id
            ].items_given_to_self["coins"]
            last_split_agent = self.state.previous_splits[agent_id].items_given_to_self[
                "coins"
            ]
        obs = TrustAndSplitObs(
            round_nb=self.state.round_nb,
            last_message=self.state.last_message,
            quota_messages_per_agent_per_round=self.quota_messages_per_agent_per_round,
            current_agent=self.state.current_agent,
            other_agent=self.agent_id_to_name[other_id],
            quantities={"coins": 10},
            item_types=self.item_types,
            value=self.state.values[agent_id],
            split_phase=self.state.split_phase,
            last_split_agent=last_split_agent,
            last_value_agent=last_value_agent,
            last_points_agent=last_points_agent,
            last_split_coagent=last_split_coagent,
            last_value_coagent=last_value_coagent,
            last_points_coagent=last_points_coagent,
        )
        return obs

    def reset(self):
        start_agent = self.agent_ids[self._starting_agent_index]
        values = self._sample_values()
        self.state = TrustAndSplitState(
            round_nb=0,
            last_message="",
            current_agent=start_agent,
            quantities={"coins": 10},
            values=values,
            previous_values=None,
            splits={aid: None for aid in self.agent_ids},
            nb_messages_sent={aid: 0 for aid in self.agent_ids},
            split_phase=False,
            previous_splits=None,
            previous_points=None,
        )
        return self.get_obs()
