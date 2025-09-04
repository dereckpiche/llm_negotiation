import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from mllm.markov_games.negotiation.nego_simulation import (
    NegotiationObs,
    NegotiationSimulation,
    NegotiationState,
    Split,
    compute_tas_style_rewards,
)

AgentId = str


@dataclass
class NoPressState(NegotiationState):
    pass


@dataclass
class NoPressObs(NegotiationObs):
    other_value: float


class NoPressSimulation(NegotiationSimulation):
    def __init__(
        self,
        deterministic: bool,
        *args,
        **kwargs,
    ):
        self.deterministic = deterministic
        super().__init__(*args, **kwargs)

    def _sample_values(self) -> Dict[AgentId, float]:
        v = float(int(self.rng.choice([1, 10])))
        return {self.agent_ids[0]: v, self.agent_ids[1]: 10.0 if v == 1.0 else 1.0}

    def set_new_round_of_variant(self):
        self.state.previous_values = copy.deepcopy(self.state.values)
        self.state.quantities = {"coins": 10.0}
        if self.deterministic:
            self.state.values = {
                aid: 1.0 if aid == self.state.current_agent else 10.0
                for aid in self.agent_ids
            }
        else:
            self.state.values = self._sample_values()
        self.state.split_phase = True

    def get_info_of_variant(
        self, state: NegotiationState, actions: Dict[AgentId, Any]
    ) -> Dict[str, Any]:
        return {
            "quantities": copy.deepcopy(state.quantities),
            "values": copy.deepcopy(state.values),
            "splits": copy.deepcopy(state.splits),
        }

    def get_rewards(self, splits: Dict[AgentId, Split]) -> Dict[AgentId, float]:
        return compute_tas_style_rewards(
            self.agent_ids, self.state.values, splits, 10.0
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
        obs = NoPressObs(
            round_nb=self.state.round_nb,
            last_message="",
            quota_messages_per_agent_per_round=self.quota_messages_per_agent_per_round,
            current_agent=self.state.current_agent,
            other_agent=other_id,
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
            other_value=self.state.values[other_id],
        )
        return obs

    def reset(self):
        start_agent = self.agent_ids[self._starting_agent_index]
        if self.deterministic:
            values = {
                aid: 1.0 if aid == start_agent else 10.0 for aid in self.agent_ids
            }
        else:
            values = self._sample_values()
        self.state = NoPressState(
            round_nb=0,
            last_message="",
            current_agent=start_agent,
            quantities={"coins": 10.0},
            values=values,
            previous_values=None,
            splits={aid: None for aid in self.agent_ids},
            nb_messages_sent={aid: 0 for aid in self.agent_ids},
            split_phase=True,
            previous_splits=None,
            previous_points=None,
        )
        return self.get_obs()
