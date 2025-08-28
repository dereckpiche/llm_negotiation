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

from mllm.markov_games.negotiation.nego_simulation import (
    Message,
    NegotiationObs,
    NegotiationSimulation,
    NegotiationState,
    Split,
    compute_tas_style_rewards,
)
from mllm.markov_games.rollout_tree import SimulationStepLog

AgentId = str


def _get_rps_winner(
    hand1: Literal["rock", "paper", "scissors"],
    hand2: Literal["rock", "paper", "scissors"],
) -> Literal["rock", "paper", "scissors"]:
    """Determine winner of rock-paper-scissors between two hands."""
    if hand1 == hand2:
        raise ValueError("Hands should be different")
    if (
        (hand1 == "rock" and hand2 == "scissors")
        or (hand1 == "paper" and hand2 == "rock")
        or (hand1 == "scissors" and hand2 == "paper")
    ):
        return hand1
    else:
        return hand2


@dataclass
class TrustAndSplitRPSState(NegotiationState):
    hands: Dict[
        AgentId, Literal["rock", "paper", "scissors"]
    ]  # rock, paper, or scissors
    previous_hands: Dict[AgentId, Literal["rock", "paper", "scissors"]] | None


@dataclass
class TrustAndSplitRPSObs(NegotiationObs):
    hand: Literal["rock", "paper", "scissors"]
    last_hand_agent: Literal["rock", "paper", "scissors"] | None
    last_hand_coagent: Literal["rock", "paper", "scissors"] | None


class TrustAndSplitRPSSimulation(NegotiationSimulation):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def _sample_hands_and_values(
        self,
    ) -> Tuple[Dict[AgentId, str], Dict[AgentId, float]]:
        # Assign different hands to each agent
        hands = ["rock", "paper", "scissors"]
        hand1, hand2 = self.rng.choice(hands, size=2, replace=False)

        agent_hands = {self.agent_ids[0]: hand1, self.agent_ids[1]: hand2}

        # Determine winner and assign values
        winner = _get_rps_winner(hand1, hand2)
        values = {}
        for agent_id in self.agent_ids:
            if agent_hands[agent_id] == winner:
                values[agent_id] = 10.0  # Winner gets value 10
            else:
                values[agent_id] = 1.0  # Loser gets value 1

        return agent_hands, values

    def set_new_round_of_variant(self):
        self.state.previous_values = copy.deepcopy(self.state.values)
        self.state.previous_hands = copy.deepcopy(self.state.hands)
        new_hands, new_values = self._sample_hands_and_values()
        self.state.hands = new_hands
        self.state.values = new_values
        # Quantities are constant in TAS
        self.state.quantities = {"coins": 10}
        self.state.split_phase = False

    def get_info_of_variant(
        self, state: NegotiationState, actions: Dict[AgentId, Any]
    ) -> Dict[str, Any]:
        return {
            "hands": copy.deepcopy(state.hands),
            "values": copy.deepcopy(state.values),
            "previous_hands": copy.deepcopy(state.previous_hands),
            "previous_values": copy.deepcopy(state.previous_values),
            "splits": copy.deepcopy(state.splits),
        }

    def get_rewards(self, splits: Dict[AgentId, Split]) -> Dict[AgentId, float]:
        return compute_tas_style_rewards(self.agent_ids, self.state.values, splits, 10)

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id"""
        other_id = self._other(agent_id)
        last_value_coagent = (
            None
            if self.state.previous_values is None
            else self.state.previous_values.get(other_id)
        )
        last_hand_coagent = (
            None
            if self.state.previous_hands is None
            else self.state.previous_hands.get(other_id)
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
        last_hand_agent = (
            None
            if self.state.previous_hands is None
            else self.state.previous_hands.get(agent_id)
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
        obs = TrustAndSplitRPSObs(
            round_nb=self.state.round_nb,
            last_message=self.state.last_message,
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
            hand=self.state.hands[agent_id],
            last_hand_coagent=last_hand_coagent,
            last_hand_agent=last_hand_agent,
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
        self.state = TrustAndSplitRPSState(
            round_nb=0,
            last_message="",
            current_agent=start_agent,
            quantities={"coins": 10},
            values=values,
            splits={aid: None for aid in self.agent_ids},
            nb_messages_sent={aid: 0 for aid in self.agent_ids},
            previous_values=None,
            previous_splits=None,
            previous_points=None,
            split_phase=False,
            hands=hands,
            previous_hands=None,
        )
        return self.get_obs()
