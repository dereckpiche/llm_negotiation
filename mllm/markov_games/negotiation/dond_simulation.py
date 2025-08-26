import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from numpy.random import default_rng

from mllm.markov_games.rollout_tree import SimulationStepLog
from mllm.markov_games.simulation import Simulation
from mllm.markov_games.negotiation.nego_simulation import Split, NegotiationState, NegotiationObs, NegotiationSimulation
from mllm.utils.get_coagent_id import get_coagent_id


AgentId = str


@dataclass
class DealNoDealState(NegotiationState):
    item_types: List[str]
    values: Dict[AgentId, Dict[str, int]]

@dataclass
class DealNoDealObs(NegotiationObs):
    my_values: Dict[str, int]
    item_types: List[str]
    previous_values_coagent: Dict[str, int] | None


def random_partition_integer(rng, total: int, parts: int) -> List[int]:
    if parts <= 0:
        return []
    if total <= 0:
        return [0 for _ in range(parts)]
    cuts = sorted(rng.integers(0, total + 1, size=parts - 1).tolist())
    vals = []
    prev = 0
    for c in cuts + [total]:
        vals.append(c - prev)
        prev = c
    return vals

class DealNoDealSimulation(NegotiationSimulation):

    def __init__(
        self,
        item_types: List[str] = ["books", "hats", "balls"],
        *args,
        **kwargs,
    ):
        super().__init__(item_types=item_types, *args, **kwargs)
        self.reset()

    def _other(self, agent_id: AgentId) -> AgentId:
        return get_coagent_id(self.agent_ids, agent_id)

    def _sample_stock(self) -> Dict[str, int]:
        # total items between 5 and 7
        total_items = int(self.rng.integers(5, 8))
        # nonnegative per-type counts summing to total_items
        parts = random_partition_integer(self.rng, total_items, len(self.item_types))
        # allow zeros per type
        return {t: int(c) for t, c in zip(self.item_types, parts)}

    def _sample_values_pair(self) -> Dict[AgentId, Dict[str, int]]:
        # Each agent has integer non-negative values that sum to 10
        # Each item type valued by at least one agent
        # Some item type valued by both agents
        while True:
            vals_a = random_partition_integer(self.rng, 10, len(self.item_types))
            vals_b = random_partition_integer(self.rng, 10, len(self.item_types))
            a = {t: int(v) for t, v in zip(self.item_types, vals_a)}
            b = {t: int(v) for t, v in zip(self.item_types, vals_b)}
            # each item valued by at least one
            ok1 = all((a[t] > 0) or (b[t] > 0) for t in self.item_types)
            # some item valued by both
            ok2 = any((a[t] > 0) and (b[t] > 0) for t in self.item_types)
            if ok1 and ok2:
                return {self.agent_ids[0]: a, self.agent_ids[1]: b}

    def _is_valid_allocation(self, allocation: Dict[str, int], stock: Dict[str, int]) -> bool:
        for t in self.item_types:
            v = allocation.get(t)
            if v is None:
                return False
            if not isinstance(v, int):
                return False
            if v < 0 or v > int(stock.get(t, 0)):
                return False
        return True
    
    def set_new_round_of_variant(self):
        self.state.quantities = {t: 0 for t in self.item_types}

    def get_info_of_variant(self, state: NegotiationState, actions: Dict[AgentId, Any]) -> Dict[str, Any]:
        return {
            "quantities": copy.deepcopy(state.quantities),
            "values": copy.deepcopy(state.values),
            'splits': copy.deepcopy(state.splits),
        }

    def get_rewards(self, splits: Dict[AgentId, Split]) -> Dict[AgentId, float]:
        """
        Returns the rewards for each agent.
        """
        split_a = splits[self.agent_ids[0]].items_given_to_self
        split_b = splits[self.agent_ids[1]].items_given_to_self
        rewards = {self.agent_ids[0]: 0, self.agent_ids[1]: 0}
        for t in self.item_types:
            # If not complementary, return 0! 
            if not split_a[t] + split_b[t] == self.state.quantities[t]:
                return {self.agent_ids[0]: 0, self.agent_ids[1]: 0}
            rewards[self.agent_ids[0]] += split_a[t] * self.state.values[self.agent_ids[0]][t]
            rewards[self.agent_ids[1]] += split_b[t] * self.state.values[self.agent_ids[1]][t]
        return rewards

    def get_obs(self):
        return {agent_id: self.get_obs_agent(agent_id) for agent_id in self.agent_ids}

    def get_obs_agent(self, agent_id):
        other_id = self._other(agent_id)
        other_prop = self.state.proposals.get(other_id) if self.state else None
        obs = DealNoDealObs(
            round_nb=self.state.round_nb,
            last_message=self.state.last_message,
            current_agent=self.state.current_agent,
            my_values=copy.deepcopy(self.state.values[agent_id]),
            item_types=list(self.item_types),
            other_agent_proposal=copy.deepcopy(other_prop),
            proposal_phase=self.state.proposal_phase,
            quota_messages_per_agent_per_round=self.quota_messages_per_agent_per_round,
            previous_values_coagent=copy.deepcopy(self.state.values.get(self._other(agent_id), {})),
        )
        return obs

    def reset(self):
        start_agent = self.agent_ids[self._starting_agent_index]
        stock = self._sample_stock()
        values = self._sample_values_pair()
        self.state = DealNoDealState(
            round_nb=0,
            last_message="",
            current_agent=start_agent,
            item_types=list(self.item_types),
            values=values,
            proposals={aid: None for aid in self.agent_ids},
            quota_messages_per_agent_per_round=self.quota_messages_per_agent_per_round,
            messages_sent={aid: 0 for aid in self.agent_ids},
            proposal_phase=False,
        )
        return self.get_obs()


