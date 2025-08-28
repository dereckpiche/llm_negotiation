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
class DeterministicNoPressState(NegotiationState):
    pass


@dataclass
class DeterministicNoPressObs(NegotiationObs):
    my_value: float
    other_value: float


class DeterministicNoPressSimulation(NegotiationSimulation):
    def __init__(
        self,
        agent_ids: List[AgentId],
        seed: int,
        rounds_per_game: int,
        max_coins: int = 10,
        *args,
        **kwargs,
    ):
        self.max_coins = int(max_coins)
        super().__init__(
            agent_ids=agent_ids,
            seed=seed,
            nb_of_rounds=int(rounds_per_game),
            quota_messages_per_agent_per_round=0,
            nb_messages_per_agent=0,
            item_types=["coins"],
        )

    def set_new_round_of_variant(self):
        # Keep fixed values and immediately enter split phase with constant quantity
        self.state.quantities = {"coins": self.max_coins}
        self.state.split_phase = True

    def get_info_of_variant(self, state: NegotiationState, actions: Dict[AgentId, Any]) -> Dict[str, Any]:
        return {
            "quantities": copy.deepcopy(state.quantities),
            "values": copy.deepcopy(state.values),
            "splits": copy.deepcopy(state.splits),
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
        return DeterministicNoPressObs(
            round_nb=self.state.round_nb,
            last_message="",
            current_agent=self.state.current_agent,
            quantities={"coins": self.max_coins},
            value=self.state.values[agent_id],
            other_agent_split=other_split_val,
            split_phase=True,
            quota_messages_per_agent_per_round=0,
            my_value=self.state.values[agent_id],
            other_value=self.state.values[other_id],
        )

    def reset(self):
        start_agent = self.agent_ids[self._starting_agent_index]
        # Fixed deterministic values: agent 0 gets 10, agent 1 gets 1
        values = {
            self.agent_ids[0]: 10.0,
            self.agent_ids[1]: 1.0,
        }
        self.state = DeterministicNoPressState(
            round_nb=0,
            last_message="",
            current_agent=start_agent,
            quantities={"coins": self.max_coins},
            values=values,
            previous_values=None,
            splits={aid: None for aid in self.agent_ids},
            nb_messages_sent={aid: 0 for aid in self.agent_ids},
            split_phase=True,
        )
        return self.get_obs()


