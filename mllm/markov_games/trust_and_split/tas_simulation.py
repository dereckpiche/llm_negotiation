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
from typing import Any, Dict, List, Tuple

from numpy.random import default_rng

from mllm.markov_games.rollout_tree import SimulationStepLog
from mllm.markov_games.simulation import Simulation
from mllm.utils.get_coagent_id import get_coagent_id

AgentId = str


@dataclass
class Split:
    coins_given_to_self: int


@dataclass
class Message:
    message: str


def get_rewards(
    coins_to_self_0: int,
    val_0: float,
    coins_to_self_1: int,
    val_1: float,
    max_coins: int,
) -> tuple[float, float]:
    denom = max(max_coins, coins_to_self_0 + coins_to_self_1)
    q0 = float(max_coins) * float(coins_to_self_0) / float(denom)
    q1 = float(max_coins) * float(coins_to_self_1) / float(denom)
    r0 = q0 * float(val_0)
    r1 = q1 * float(val_1)
    return r0, r1


def _get_rps_winner(hand1: str, hand2: str) -> str:
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
class TrustAndSplitState:
    round_nb: int
    last_message: str
    current_agent: AgentId
    values: Dict[AgentId, float]
    previous_values: Dict[AgentId, float] | None
    hands: Dict[AgentId, str]  # rock, paper, or scissors
    previous_hands: Dict[AgentId, str] | None
    splits: Dict[AgentId, Split | None]
    messages_sent: Dict[AgentId, int]
    split_phase: bool = False


@dataclass
class TrustAndSplitObs:
    round_nb: int
    last_message: str
    current_agent: AgentId
    last_value_coagent: float | None
    value: float
    other_agent_split: int | None
    hand: str  # rock, paper, or scissors
    last_hand_coagent: str | None
    split_phase: bool = False

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
        self.nb_messages_per_agent = int(nb_messages_per_agent)
        self.max_coins = int(max_coins)
        # Unused but kept for compatibility with earlier drafts
        self.no_smooth_split = (
            bool(no_smooth_split) if no_smooth_split is not None else False
        )
        self.item_types = item_types or ["coins"]
        self.state: TrustAndSplitState | None = None
        self._starting_agent_index = 0
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

    def _other(self, agent_id: AgentId) -> AgentId:
        return get_coagent_id(self.agent_ids, agent_id)

    def step(self, actions: Any) -> Tuple[bool, SimulationStepLog]:
        """
        Returns terminated, step_log
        """
        assert self.state is not None
        current_agent = self.state.current_agent
        a0, a1 = self.agent_ids[0], self.agent_ids[1]
        action = actions.get(current_agent)

        # Split phase: require both splits in the same timestep
        if self.state.split_phase:
            action_a0 = actions.get(a0)
            action_a1 = actions.get(a1)
            have_both_splits = isinstance(action_a0, Split) and isinstance(
                action_a1, Split
            )
            if not have_both_splits:
                rewards = {agent_id: 0.0 for agent_id in self.agent_ids}
                return False, SimulationStepLog(
                    rewards=rewards, info={"type": "waiting_for_splits"}
                )

            # Record splits
            self.state.splits[a0] = action_a0
            self.state.splits[a1] = action_a1

            # Compute rewards and end round
            coins_to_self_0 = int(
                self.state.splits[a0].coins_given_to_self
                if self.state.splits[a0] is not None
                else 0
            )
            coins_to_self_1 = int(
                self.state.splits[a1].coins_given_to_self
                if self.state.splits[a1] is not None
                else 0
            )
            val_0 = float(self.state.values[a0])
            val_1 = float(self.state.values[a1])
            sums_to_max = (coins_to_self_0 + coins_to_self_1) == self.max_coins
            r0, r1 = get_rewards(
                coins_to_self_0, val_0, coins_to_self_1, val_1, self.max_coins
            )
            rewards = {a0: r0, a1: r1}

            # Prepare next round
            self.state.previous_values = copy.deepcopy(self.state.values)
            self.state.previous_hands = copy.deepcopy(self.state.hands)
            new_hands, new_values = self._sample_hands_and_values()
            self.state.hands = new_hands
            self.state.values = new_values
            self.state.round_nb += 1
            self.state.last_message = ""
            self.state.split_phase = False
            self.state.splits = {aid: None for aid in self.agent_ids}
            self.state.messages_sent = {aid: 0 for aid in self.agent_ids}
            # Alternate starting agent
            self._starting_agent_index = 1 - self._starting_agent_index
            self.state.current_agent = self.agent_ids[self._starting_agent_index]

            done = self.state.round_nb >= self.rounds_per_game
            return done, SimulationStepLog(
                rewards=rewards,
                info=SplitsLog(
                    sums_to_max_coins=sums_to_max,
                    num_coins=self.max_coins,
                    values={a0: val_0, a1: val_1},
                    hands={a0: self.state.previous_hands[a0], a1: self.state.previous_hands[a1]},
                    coins_given_to_self={a0: coins_to_self_0, a1: coins_to_self_1},
                ),
            )

        # Message phase
        if isinstance(action, Message):
            self.state.last_message = action.message
            self.state.messages_sent[current_agent] += 1

            # Move turn to other agent
            self.state.current_agent = self._other(current_agent)

            # If both agents have reached their message quota, enter split phase
            if all(
                self.state.messages_sent[aid] >= self.nb_messages_per_agent
                for aid in self.agent_ids
            ):
                self.state.split_phase = True

            rewards = {agent_id: 0.0 for agent_id in self.agent_ids}
            return False, SimulationStepLog(rewards=rewards, info={"type": "message"})

        # Splits are only valid during split_phase; otherwise invalid action
        if isinstance(action, Split):
            raise Exception(
                "Split received outside of split_phase. Agents must message until split_phase begins."
            )

        raise Exception("Invalid action type for TrustAndSplitSimulation.")

    def get_obs(self):
        """Returns all agent observations in dict"""
        return {agent_id: self.get_obs_agent(agent_id) for agent_id in self.agent_ids}

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
