import copy
import random
from typing import Any, Dict, List, Optional, Tuple
from mllm.markov_games.markov_game import Simulation
import numpy as np
from dataclasses import dataclass
from mllm.utils.get_coagent_id import get_coagent_id

@dataclass
class IPDState:
    """
    State of the Iterated Prisoner's Dilemma game.
    """
    round_nb: int = 0
    done: bool = False
    last_moves: Dict[str, str] | None = None

@dataclass
class IPDObs:
    """
    Observation in Iterated Prisoner's Dilemma game.
    """
    round_nb: int
    last_coagent_move: str | None

class IPD(Simulation):
    """
    Iterated Prisoner's Dilemma simulation following the standard.

    In each round of the game, two agents simultaneously choose to either cooperate (C) or defect (D).
    The payoffs are as follows:
    - If both cooperate: Both receive the "reward" (usually 3 points)
    - If both defect: Both receive the "punishment" (usually 1 point)
    - If one cooperates and one defects: The defector receives the "temptation" (usually 5 points)
      and the cooperator receives the "sucker" payoff (usually 0 points)

    The game is played for a specified number of rounds.
    """
    d = 5
    def __init__(
        self,
        agent_ids: List[str],
        seed: int,
        rounds_per_game: int,
        reward: float,  # Both cooperate
        punishment: float, # Both defect
        temptation: float,  # Defector's reward when other cooperates
        sucker: float,  # Cooperator's reward when other defects
        cooperate_actions: List[str],
        defect_actions: List[str],
    ):
        self.agent_ids = agent_ids
        self.seed =seed
        self.rounds_per_game =rounds_per_game
        self.reward =reward
        self.punishment =punishment
        self.temptation =temptation
        self.sucker =sucker
        self.cooperate_actions =cooperate_actions
        self.defect_actions =defect_actions
        self.gibberish_action = "GIBBERISH"
        self.state = IPDState()

    def step(
        self, actions: Dict[str, str]
    ) -> Tuple[bool, dict]:
        """
        Take a step in the environment using the provided actions.
        Here, the observations are just the states of the game.

        Args:
            actions (dict): A dictionary where keys are agent identifiers and values are actions ('C' or 'D').

        Returns:
            observations (dict): A dictionary where keys are agent identifiers and values are observations.
            done (bool): Whether the episode has ended.
            info (dict): Additional information about the environment.
        """

        # Calculate rewards based on the prisoner's dilemma payoff matrix
        round_rewards = {}
        agent0_action = actions[self.agent_ids[0]]
        agent1_action = actions[self.agent_ids[1]]

        if (
            agent0_action in self.cooperate_actions

            and agent1_action in self.cooperate_actions
        ):
            # Both cooperate
            round_rewards[self.agent_ids[0]] = self.reward
            round_rewards[self.agent_ids[1]] = self.reward
        elif (
            agent0_action in self.defect_actions
            and agent1_action in self.defect_actions
        ):
            # Both defect
            round_rewards[self.agent_ids[0]] = self.punishment
            round_rewards[self.agent_ids[1]] = self.punishment
        elif (
            agent0_action in self.cooperate_actions
            and agent1_action in self.defect_actions
        ):
            # Alice cooperates, Bob defects
            round_rewards[self.agent_ids[0]] = self.sucker
            round_rewards[self.agent_ids[1]] = self.temptation
        elif (
            agent0_action in self.defect_actions
            and agent1_action in self.cooperate_actions
        ):
            # Alice defects, Bob cooperates
            round_rewards[self.agent_ids[0]] = self.temptation
            round_rewards[self.agent_ids[1]] = self.sucker
        else:
            # TODO: find clean solution for this
            # If any of the agents outputs Gibberish set rewards to 0
            round_rewards[self.agent_ids[0]] = 0
            round_rewards[self.agent_ids[1]] = 0

        # Update game state
        self.state.round_nb += 1
        self.state.last_moves = copy.deepcopy(actions)
        done = self.state.round_nb >= self.rounds_per_game
        info = {"rewards": round_rewards}

        return done, info

    def get_obs(self):
        """ Returns all agent observations in dict
        Returns:
            observations
        """
        observations = {}
        for agent_id in self.agent_ids:
            observations[agent_id] = self.get_obs_agent(agent_id)
        return observations

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        if self.state.last_moves != None:
            other_id = get_coagent_id(self.agent_ids, agent_id)
            last_coagent_move = self.state.last_moves[other_id]
        else:
            last_coagent_move = None
        obs = IPDObs(
            round_nb = self.state.round_nb,
            last_coagent_move=last_coagent_move
        )
        return obs

    def get_obs_size(self):
        """ Returns the shape of the observation """
        pass

    def get_state(self):
        return self.state

    def get_state_size(self):
        """ Returns the shape of the state"""
        pass

    def get_avail_actions(self):
        pass

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        pass

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        pass

    def reset(self):
        """Returns initial observations and states"""
        self.state = IPDState()
        return self.get_obs()

    def render(self):
        pass

    def close(self):
        pass

    # def seed(self):
    #     pass

    def save_replay(self):
        pass

    def get_env_info(self):
        pass
