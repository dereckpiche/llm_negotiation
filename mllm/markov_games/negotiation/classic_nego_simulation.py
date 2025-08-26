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
class ClassicState(NegotiationState):
    pass


@dataclass
class ClassicObs(NegotiationObs):
    value: float  # override to ensure present


class ClassicSimulation(NegotiationSimulation):
    def __init__(
        self,
        agent_ids: List[AgentId],
        seed: int,
        rounds_per_game: int,
        quota_messages_per_agent_per_round: int,
        nb_messages_per_agent: int = 1,
        max_coins: int = 10,
        *args,
        **kwargs,
    ):
        self.max_coins = int(max_coins)
        super().__init__(
            agent_ids=agent_ids,
            seed=seed,
            nb_of_rounds=int(rounds_per_game),
            quota_messages_per_agent_per_round=int(quota_messages_per_agent_per_round),
            nb_messages_per_agent=int(nb_messages_per_agent),
            item_types=["coins"],
        )

    def _sample_values(self) -> Dict[AgentId, float]:
        # Independent random per-coin values between 1 and 20 (inclusive)
        return {aid: float(int(self.rng.integers(1, 21))) for aid in self.agent_ids}

    def set_new_round_of_variant(self):
        self.state.previous_values = copy.deepcopy(self.state.values)
        self.state.values = self._sample_values()
        self.state.quantities = {"coins": self.max_coins}

    def get_info_of_variant(self, state: NegotiationState, actions: Dict[AgentId, Any]) -> Dict[str, Any]:
        return {
            "values": copy.deepcopy(state.values),
            "previous_values": copy.deepcopy(state.previous_values),
            "splits": copy.deepcopy(state.splits),
            "quantities": copy.deepcopy(state.quantities),
        }

    def get_rewards(self, splits: Dict[AgentId, Split]) -> Dict[AgentId, float]:
        return compute_tas_style_rewards(self.agent_ids, self.state.values, splits, self.max_coins)

    def get_obs(self):
        return {agent_id: self.get_obs_agent(agent_id) for agent_id in self.agent_ids}

    def get_obs_agent(self, agent_id):
        other_id = self.agent_ids[1] if agent_id == self.agent_ids[0] else self.agent_ids[0]
        other_split_val = None
        if self.state.splits.get(other_id) is not None:
            other_split_val = self.state.splits[other_id].items_given_to_self.get("coins", 0)
        return ClassicObs(
            round_nb=self.state.round_nb,
            last_message=self.state.last_message,
            current_agent=self.state.current_agent,
            quantities={"coins": self.max_coins},
            value=self.state.values[agent_id],
            other_agent_split=other_split_val,
            split_phase=self.state.split_phase,
            quota_messages_per_agent_per_round=self.quota_messages_per_agent_per_round,
        )

    def reset(self):
        start_agent = self.agent_ids[self._starting_agent_index]
        values = self._sample_values()
        self.state = ClassicState(
            round_nb=0,
            last_message="",
            current_agent=start_agent,
            quantities={"coins": self.max_coins},
            values=values,
            previous_values=None,
            splits={aid: None for aid in self.agent_ids},
            nb_messages_sent={aid: 0 for aid in self.agent_ids},
            split_phase=False,
        )
        return self.get_obs()


