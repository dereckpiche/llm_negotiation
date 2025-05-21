import copy
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class IPDGameState:
    """
    State of the Iterated Prisoner's Dilemma game.
    """

    def __init__(self):
        self.match_id = None
        self.group_id = None
        self.agent_ids = None
        self.number_of_rounds = None
        self.round_nb = 0
        self.rewards = []
        self.raw_actions = []
        self.actions = []
        self.done = False
        self.info = {}
        self.cooperate_actions = ["C"]
        self.defect_actions = ["D"]
        self.gibberish_action = "GIBBERISH"


class IPDEnv:
    """
    Iterated Prisoner's Dilemma environment following the MarlEnvironment standard.

    In each round of the game, two agents simultaneously choose to either cooperate (C) or defect (D).
    The payoffs are as follows:
    - If both cooperate: Both receive the "reward" (usually 3 points)
    - If both defect: Both receive the "punishment" (usually 1 point)
    - If one cooperates and one defects: The defector receives the "temptation" (usually 5 points)
      and the cooperator receives the "sucker" payoff (usually 0 points)

    The game is played for a specified number of rounds.
    """

    def __init__(
        self,
        agents: List[str],
        rng: np.random.RandomState,
        group_id: int,
        game_id: int,
        rounds_per_game: int,
        reward: float,  # Both cooperate
        punishment: float,  # Both defect
        temptation: float,  # Defector's reward when other cooperates
        sucker: float,  # Cooperator's reward when other defects
        random_seed: Optional[int] = None,
        cooperate_actions: Optional[list] = None,
        defect_actions: Optional[list] = None,
        gibberish_action: str = "GIBBERISH",
    ):
        """
        Initialize the Iterated Prisoner's Dilemma environment.

        Args:
            rounds_per_game: Number of rounds to play
            reward: Payoff when both agents cooperate
            punishment: Payoff when both agents defect
            temptation: Payoff for defecting when other agent cooperates
            sucker: Payoff for cooperating when other agent defects
            seed: Random seed for reproducibility
        """
        self.reward = reward
        self.punishment = punishment
        self.temptation = temptation
        self.sucker = sucker
        self.rounds_per_game = rounds_per_game

        self.agent_ids = agents
        self.player_0_id = agents[0]
        self.player_1_id = agents[1]
        self.match_id = game_id
        self.group_id = group_id
        self.gibberish_action = gibberish_action
        self.cooperate_actions = (
            cooperate_actions
            if cooperate_actions is not None
            else ["C", "<Cooperate>", "<A>"]
        )
        self.defect_actions = (
            defect_actions if defect_actions is not None else ["D", "<Defect>", "<B>"]
        )
        self.reset()

    def reset(self):
        """
        Reset the environment to an initial state and return the initial observation.
        """
        self.state = IPDGameState()
        self.state.agent_ids = self.agent_ids
        self.state.match_id = self.match_id
        self.state.group_id = self.group_id
        self.state.number_of_rounds = self.rounds_per_game
        # Propagate action string settings to state
        self.state.cooperate_actions = self.cooperate_actions
        self.state.defect_actions = self.defect_actions
        self.state.gibberish_action = self.gibberish_action
        return {self.player_0_id: self.state, self.player_1_id: self.state}

    def step(
        self, actions: Dict[str, str]
    ) -> Tuple[Dict[str, Dict[str, Any]], bool, Dict[str, Any]]:
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
        p0_action = actions["Alice"]
        p1_action = actions["Bob"]
        # import pdb; pdb.set_trace()

        self.state.raw_actions.append(copy.deepcopy(actions))
        actions = copy.deepcopy(actions)

        if (
            p0_action in self.state.cooperate_actions
            and p1_action in self.state.cooperate_actions
        ):
            # Both cooperate
            round_rewards[self.player_0_id] = self.reward
            round_rewards[self.player_1_id] = self.reward
        elif (
            p0_action in self.state.defect_actions
            and p1_action in self.state.defect_actions
        ):
            # Both defect
            round_rewards[self.player_0_id] = self.punishment
            round_rewards[self.player_1_id] = self.punishment
        elif (
            p0_action in self.state.cooperate_actions
            and p1_action in self.state.defect_actions
        ):
            # Alice cooperates, Bob defects
            round_rewards[self.player_0_id] = self.sucker
            round_rewards[self.player_1_id] = self.temptation
        elif (
            p0_action in self.state.defect_actions
            and p1_action in self.state.cooperate_actions
        ):
            # Alice defects, Bob cooperates
            round_rewards[self.player_0_id] = self.temptation
            round_rewards[self.player_1_id] = self.sucker
        else:
            # TODO: find clean solution for this
            # If any of the players outputs Gibberish set rewards to 0
            round_rewards[self.player_0_id] = 0
            round_rewards[self.player_1_id] = 0

        # Update game state
        self.state.round_nb += 1
        self.state.rewards.append(round_rewards)
        self.state.actions.append(actions)

        done = self.state.round_nb >= self.rounds_per_game

        return {self.player_0_id: self.state, self.player_1_id: self.state}, done, {}

    def get_log_info(self) -> Dict[str, Any]:
        """
        Get additional information about the environment for logging purposes.

        Returns:
            log_info (dict): Information about the environment required to log the game.
        """
        return self.state

    def render(self) -> str:
        """
        Render the current state of the environment as a string.

        Returns:
            str: A string representation of the current state.
        """
        pass

    def close(self) -> None:
        """
        Perform any necessary cleanup.
        """
        pass
