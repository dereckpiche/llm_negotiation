"""
Trust-and-Split simulation.

This environment models a simple bargaining game over 10 coins with messaging.
Agents are assigned rock/paper/scissors hands, with the winner getting value 10 per coin
and the loser getting value 1 per coin. Agents alternate sending messages for a fixed 
number of turns per round and then each submits a split proposal indicating how many 
coins they keep for themselves. Rewards are proportional if the proposed totals exceed 10.
"""

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Tuple

from numpy.random import default_rng

from mllm.markov_games.rollout_tree import SimulationStepLog
from mllm.markov_games.simulation import Simulation
from mllm.markov_games.negotiation.nego_simulation import Split, Message, NegotiationState, NegotiationObs

AgentId = str


def _get_rps_winner(hand1: Literal["rock", "paper", "scissors"], hand2: Literal["rock", "paper", "scissors"]) -> Literal["rock", "paper", "scissors"]:
    """Determine winner of rock-paper-scissors between two hands."""
    if hand1 == hand2:
        raise ValueError("Hands should be different")
    if (hand1 == "rock" and hand2 == "scissors") or \
       (hand1 == "paper" and hand2 == "rock") or \
       (hand1 == "scissors" and hand2 == "paper"):
        return hand1
    else:
        return hand2


@dataclass
class TrustAndSplitState(NegotiationState):
    hands: Dict[AgentId, Literal["rock", "paper", "scissors"]]  # rock, paper, or scissors
    previous_hands: Dict[AgentId, Literal["rock", "paper", "scissors"]] | None

@dataclass
class TrustAndSplitObs(NegotiationObs):
    last_value_coagent: float | None
    hand: Literal["rock", "paper", "scissors"]
    last_hand_coagent: Literal["rock", "paper", "scissors"] | None

@dataclass
class SplitsLog:
    sums_to_max_coins: bool
    num_coins: int
    values: Dict[AgentId, float]
    hands: Dict[AgentId, str]
    coins_given_to_self: Dict[AgentId, int]
    

class TrustAndSplitSimulation(Simulation):
    def __init__(
        self,
        agent_ids: List[AgentId],
        seed: int,
        rounds_per_game: int,
        quota_messages_per_agent_per_round: int,
        nb_messages_per_agent: int = 1,
        max_coins: int = 10,
        no_smooth_split: bool | None = None,
        item_types: List[str] | None = None,
        *args,
        **kwargs,
    ):
        self.seed = seed
        self.rng = default_rng(self.seed)
        self.agent_ids = list(agent_ids)
        self.rounds_per_game = int(rounds_per_game)
        self.quota_messages_per_agent_per_round = int(quota_messages_per_agent_per_round)
        self.nb_messages_per_agent = int(nb_messages_per_agent)
        self.max_coins = int(max_coins)
        # Unused but kept for compatibility with earlier drafts
        self.no_smooth_split = (
            bool(no_smooth_split) if no_smooth_split is not None else False
        )
        self.item_types = item_types or ["coins"]
        self.state: TrustAndSplitState | None = None
        self._starting_agent_index = self.rng.choice([0, 1])
        self.reset()

    def _sample_hands_and_values(self) -> Tuple[Dict[AgentId, str], Dict[AgentId, float]]:
        # Assign different hands to each agent
        hands = ["rock", "paper", "scissors"]
        hand1, hand2 = self.rng.choice(hands, size=2, replace=False)
        
        agent_hands = {
            self.agent_ids[0]: hand1,
            self.agent_ids[1]: hand2
        }
        
        # Determine winner and assign values
        winner = _get_rps_winner(hand1, hand2)
        values = {}
        for agent_id in self.agent_ids:
            if agent_hands[agent_id] == winner:
                values[agent_id] = 10.0  # Winner gets value 10
            else:
                values[agent_id] = 1.0   # Loser gets value 1
                
        return agent_hands, values

    def set_new_round_of_variant(self):
        self.state.previous_values = copy.deepcopy(self.state.values)
        self.state.previous_hands = copy.deepcopy(self.state.hands)
        new_hands, new_values = self._sample_hands_and_values()
        self.state.hands = new_hands
        self.state.values = new_values

    def get_info_of_variant(self, state: NegotiationState, actions: Dict[AgentId, Any]) -> Dict[str, Any]:
        return {
            "hands": copy.deepcopy(state.hands),
            "values": copy.deepcopy(state.values),
            "previous_hands": copy.deepcopy(state.previous_hands),
            "previous_values": copy.deepcopy(state.previous_values),
            "splits": copy.deepcopy(state.splits),
        }

    def get_obs(self):
        """Returns all agent observations in dict"""
        return {agent_id: self.get_obs_agent(agent_id) for agent_id in self.agent_ids}

    def get_rewards(self, splits: Dict[AgentId, Split]) -> Dict[AgentId, float]:
        a0, a1 = self.agent_ids[0], self.agent_ids[1]
        coins_to_self_0 = int(splits[a0].items_given_to_self.values()[0] if splits[a0] is not None else 0)
        coins_to_self_1 = int(splits[a1].items_given_to_self.values()[0] if splits[a1] is not None else 0)
        denom = max(self.max_coins, coins_to_self_0 + coins_to_self_1)
        q0 = float(self.max_coins) * float(coins_to_self_0) / float(denom)
        q1 = float(self.max_coins) * float(coins_to_self_1) / float(denom)
        r0 = q0 * float(self.state.values[a0])
        r1 = q1 * float(self.state.values[a1])
        return {a0: r0, a1: r1}

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id"""
        other_id = self._other(agent_id)
        last_value = (
            None
            if self.state.previous_values is None
            else self.state.previous_values.get(other_id)
        )
        last_hand = (
            None
            if self.state.previous_hands is None
            else self.state.previous_hands.get(other_id)
        )
        other_split_val = None
        if self.state.splits.get(other_id) is not None:
            other_split_val = self.state.splits[other_id].coins_given_to_self
        obs = TrustAndSplitObs(
            round_nb=self.state.round_nb,
            last_message=self.state.last_message,
            current_agent=self.state.current_agent,
            last_value_coagent=last_value,
            value=self.state.values[agent_id],
            other_agent_split=other_split_val,
            quota_messages_per_agent_per_round=self.quota_messages_per_agent_per_round,
            hand=self.state.hands[agent_id],
            last_hand_coagent=last_hand,
            split_phase=self.state.split_phase,
        )
        return obs

    def get_state(self):
        return self.state

    def get_safe_copy(self):
        """Return a safe copy of the simulation."""
        simulation_copy = copy.copy(self)
        simulation_copy.state = copy.deepcopy(self.state)
        return simulation_copy

    def reset(self):
        """Initialize and return initial observations"""
        # Decide starting agent alternating across resets for determinism
        start_agent = self.agent_ids[self._starting_agent_index]
        hands, values = self._sample_hands_and_values()
        self.state = TrustAndSplitState(
            round_nb=0,
            last_message="",
            current_agent=start_agent,
            values=values,
            previous_values=None,
            hands=hands,
            previous_hands=None,
            splits={aid: None for aid in self.agent_ids},
            messages_sent={aid: 0 for aid in self.agent_ids},
            split_phase=False,
        )
        return self.get_obs()
