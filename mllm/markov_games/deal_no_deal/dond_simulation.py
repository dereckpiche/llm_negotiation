import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from numpy.random import default_rng

from mllm.markov_games.rollout_tree import SimulationStepLog
from mllm.markov_games.simulation import Simulation
from mllm.utils.get_coagent_id import get_coagent_id


AgentId = str


@dataclass
class Message:
    message: str


@dataclass
class Proposal:
    no_deal: bool
    allocation_to_self: Dict[str, int] | None = None


@dataclass
class DealNoDealState:
    round_nb: int
    last_message: str
    current_agent: AgentId
    item_types: List[str]
    stock: Dict[str, int]
    values: Dict[AgentId, Dict[str, int]]
    proposals: Dict[AgentId, Proposal | None]
    quota_messages_per_agent_per_round: int
    messages_sent: Dict[AgentId, int]
    proposal_phase: bool = False


@dataclass
class DealNoDealObs:
    round_nb: int
    last_message: str
    current_agent: AgentId
    my_values: Dict[str, int]
    item_types: List[str]
    stock: Dict[str, int]
    other_agent_proposal: Proposal | None
    proposal_phase: bool
    quota_messages_per_agent_per_round: int
    previous_values_coagent: Dict[str, int] | None

@dataclass
class DealLog:
    accepted: bool
    no_deal: bool
    item_types: List[str]
    stock: Dict[str, int]
    values: Dict[AgentId, Dict[str, int]]
    proposals: Dict[AgentId, Proposal]
    final_allocation_to_self: Dict[AgentId, Dict[str, int]] | None


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


class DealNoDealSimulation(Simulation):
    def __init__(
        self,
        agent_ids: List[AgentId],
        seed: int,
        rounds_per_game: int,
        quota_messages_per_agent_per_round: int = 1,
        item_types: List[str] | None = None,
    ):
        self.seed = seed
        self.rng = default_rng(self.seed)
        self.agent_ids = list(agent_ids)
        self.rounds_per_game = int(rounds_per_game)
        self.quota_messages_per_agent_per_round = int(quota_messages_per_agent_per_round)
        self.item_types = item_types or ["books", "hats", "balls"]
        self.state: DealNoDealState | None = None
        self._starting_agent_index = 0
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

    def step(self, actions: Any) -> Tuple[bool, SimulationStepLog]:
        assert self.state is not None
        current_agent = self.state.current_agent
        a0, a1 = self.agent_ids[0], self.agent_ids[1]
        action = actions.get(current_agent)

        if self.state.proposal_phase:
            p0 = actions.get(a0)
            p1 = actions.get(a1)
            have_both = isinstance(p0, Proposal) and isinstance(p1, Proposal)
            if not have_both:
                rewards = {agent_id: 0.0 for agent_id in self.agent_ids}
                return False, SimulationStepLog(
                    rewards=rewards, info={"type": "waiting_for_proposals"}
                )

            self.state.proposals[a0] = p0
            self.state.proposals[a1] = p1

            accepted = False
            no_deal = False
            final_alloc_to_self: Dict[AgentId, Dict[str, int]] | None = None

            if p0.no_deal or p1.no_deal:
                no_deal = True
            else:
                alloc0 = p0.allocation_to_self or {}
                alloc1 = p1.allocation_to_self or {}
                v0_ok = self._is_valid_allocation(alloc0, self.state.stock)
                v1_ok = self._is_valid_allocation(alloc1, self.state.stock)
                if v0_ok and v1_ok:
                    complement = all((alloc0.get(t, 0) + alloc1.get(t, 0)) == self.state.stock[t] for t in self.item_types)
                    if complement:
                        accepted = True
                        final_alloc_to_self = {a0: alloc0, a1: alloc1}

            if accepted:
                r0 = sum(final_alloc_to_self[a0][t] * self.state.values[a0][t] for t in self.item_types)
                r1 = sum(final_alloc_to_self[a1][t] * self.state.values[a1][t] for t in self.item_types)
                rewards = {a0: float(r0), a1: float(r1)}
            else:
                rewards = {a0: 0.0, a1: 0.0}

            # prepare info using current round's parameters
            info_stock = copy.deepcopy(self.state.stock)
            info_values = copy.deepcopy(self.state.values)

            # next round
            new_stock = self._sample_stock()
            new_values = self._sample_values_pair()
            self.state.round_nb += 1
            self.state.last_message = ""
            self.state.proposal_phase = False
            self.state.proposals = {aid: None for aid in self.agent_ids}
            self.state.messages_sent = {aid: 0 for aid in self.agent_ids}
            self.state.quota_messages_per_agent_per_round = self.quota_messages_per_agent_per_round
            self.state.stock = new_stock
            self.state.values = new_values
            self._starting_agent_index = 1 - self._starting_agent_index
            self.state.current_agent = self.agent_ids[self._starting_agent_index]

            done = self.state.round_nb >= self.rounds_per_game
            return done, SimulationStepLog(
                rewards=rewards,
                info=DealLog(
                    accepted=accepted,
                    no_deal=no_deal,
                    item_types=list(self.item_types),
                    stock=info_stock,
                    values=info_values,
                    proposals={a0: p0, a1: p1},
                    final_allocation_to_self=final_alloc_to_self,
                ),
            )

        if isinstance(action, Message):
            self.state.last_message = action.message
            self.state.messages_sent[current_agent] += 1
            self.state.current_agent = self._other(current_agent)
            if all(self.state.messages_sent[aid] >= self.quota_messages_per_agent_per_round for aid in self.agent_ids):
                self.state.proposal_phase = True
            rewards = {agent_id: 0.0 for agent_id in self.agent_ids}
            return False, SimulationStepLog(rewards=rewards, info={"type": "message"})

        if isinstance(action, Proposal):
            raise Exception("Proposal received outside of proposal_phase")

        raise Exception("Invalid action type for DealNoDealSimulation")

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
            stock=copy.deepcopy(self.state.stock),
            other_agent_proposal=copy.deepcopy(other_prop),
            proposal_phase=self.state.proposal_phase,
            quota_messages_per_agent_per_round=self.quota_messages_per_agent_per_round,
            previous_values_coagent=copy.deepcopy(self.state.values.get(self._other(agent_id), {})),
        )
        return obs

    def get_state(self):
        return self.state

    def get_safe_copy(self):
        simulation_copy = copy.copy(self)
        simulation_copy.state = copy.deepcopy(self.state)
        return simulation_copy

    def reset(self):
        start_agent = self.agent_ids[self._starting_agent_index]
        stock = self._sample_stock()
        values = self._sample_values_pair()
        self.state = DealNoDealState(
            round_nb=0,
            last_message="",
            current_agent=start_agent,
            item_types=list(self.item_types),
            stock=stock,
            values=values,
            proposals={aid: None for aid in self.agent_ids},
            quota_messages_per_agent_per_round=self.quota_messages_per_agent_per_round,
            messages_sent={aid: 0 for aid in self.agent_ids},
            proposal_phase=False,
        )
        return self.get_obs()


