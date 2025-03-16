from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import random


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
        rounds_per_game: int = 10,
        reward: float = 3.0,           # Both cooperate
        punishment: float = 1.0,       # Both defect
        temptation: float = 5.0,       # Defector's reward when other cooperates
        sucker: float = 0.0,           # Cooperator's reward when other defects
        random_seed: Optional[int] = None,
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
        self.rounds_per_game = rounds_per_game
        self.reward = reward
        self.punishment = punishment
        self.temptation = temptation
        self.sucker = sucker
        
        # Set random seed if provided
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        # Initialize game state
        self.current_round = 0
        self.agent_ids = ["alice", "bob"]
        self.history = []
        self.total_rewards = {agent_id: 0.0 for agent_id in self.agent_ids}
        self.action_space = ["C", "D"]  # Cooperate or Defect
        
    def reset(self) -> Dict[str, Dict[str, Any]]:
        """
        Reset the environment to an initial state and return the initial observation.
        
        Returns:
            observation (dict): A dictionary where keys are agent identifiers and values are observations.
        """
        self.current_round = 0
        self.history = []
        self.total_rewards = {agent_id: 0.0 for agent_id in self.agent_ids}
        
        # Initial observation for each agent
        observations = {}
        for agent_id in self.agent_ids:
            observations[agent_id] = {
                "current_round": self.current_round,
                "rounds_per_game": self.rounds_per_game,
                "history": self.history.copy(),
                "total_reward": self.total_rewards[agent_id],
                "payoff_matrix": {
                    "reward": self.reward,
                    "punishment": self.punishment,
                    "temptation": self.temptation,
                    "sucker": self.sucker,
                }
            }
        
        return observations
    
    def step(self, actions: Dict[str, str]) -> Tuple[Dict[str, Dict[str, Any]], bool, Dict[str, Any]]:
        """
        Take a step in the environment using the provided actions.
        
        Args:
            actions (dict): A dictionary where keys are agent identifiers and values are actions ('C' or 'D').
        
        Returns:
            observations (dict): A dictionary where keys are agent identifiers and values are observations.
            done (bool): Whether the episode has ended.
            info (dict): Additional information about the environment.
        """
        # Validate actions
        for agent_id, action in actions.items():
            if action not in self.action_space:
                raise ValueError(f"Invalid action '{action}' for agent '{agent_id}'. Valid actions are {self.action_space}.")
        
        # Calculate rewards based on the prisoner's dilemma payoff matrix
        round_rewards = {}
        alice_action = actions["alice"]
        bob_action = actions["bob"]
        
        if alice_action == "C" and bob_action == "C":
            # Both cooperate
            round_rewards["alice"] = self.reward
            round_rewards["bob"] = self.reward
        elif alice_action == "D" and bob_action == "D":
            # Both defect
            round_rewards["alice"] = self.punishment
            round_rewards["bob"] = self.punishment
        elif alice_action == "C" and bob_action == "D":
            # Alice cooperates, Bob defects
            round_rewards["alice"] = self.sucker
            round_rewards["bob"] = self.temptation
        else:  # alice_action == "D" and bob_action == "C"
            # Alice defects, Bob cooperates
            round_rewards["alice"] = self.temptation
            round_rewards["bob"] = self.sucker
        
        # Update game state
        self.history.append({
            "round": self.current_round,
            "actions": actions.copy(),
            "rewards": round_rewards.copy()
        })
        
        # Update total rewards
        for agent_id in self.agent_ids:
            self.total_rewards[agent_id] += round_rewards[agent_id]
        
        # Increment round counter
        self.current_round += 1
        
        # Check if game is done
        done = self.current_round >= self.rounds_per_game
        
        # Prepare observations for next round
        observations = {}
        for agent_id in self.agent_ids:
            observations[agent_id] = {
                "current_round": self.current_round,
                "rounds_per_game": self.rounds_per_game,
                "history": self.history.copy(),
                "last_round_actions": actions.copy(),
                "last_round_reward": round_rewards[agent_id],
                "total_reward": self.total_rewards[agent_id],
                "payoff_matrix": {
                    "reward": self.reward,
                    "punishment": self.punishment,
                    "temptation": self.temptation,
                    "sucker": self.sucker,
                }
            }
        
        # Prepare info dictionary
        info = {
            "round_history": self.history.copy(),
            "total_rewards": self.total_rewards.copy(),
            "current_round": self.current_round,
        }
        
        return observations, done, info
    
    def get_log_info(self) -> Dict[str, Any]:
        """
        Get additional information about the environment for logging purposes.
        
        Returns:
            log_info (dict): Information about the environment required to log the game.
        """
        return {
            "environment": "Iterated Prisoner's Dilemma",
            "rounds_per_game": self.rounds_per_game,
            "payoff_matrix": {
                "reward": self.reward,
                "punishment": self.punishment,
                "temptation": self.temptation,
                "sucker": self.sucker,
            },
            "history": self.history.copy(),
            "total_rewards": self.total_rewards.copy(),
            "current_round": self.current_round,
        }
    
    def render(self) -> str:
        """
        Render the current state of the environment as a string.
        
        Returns:
            str: A string representation of the current state.
        """
        output = []
        output.append(f"Iterated Prisoner's Dilemma - Round {self.current_round}/{self.rounds_per_game}")
        output.append(f"Payoff Matrix: R={self.reward}, P={self.punishment}, T={self.temptation}, S={self.sucker}")
        output.append("\nHistory:")
        
        if not self.history:
            output.append("No rounds played yet.")
        else:
            output.append("Round\tAlice\tBob\tAlice Reward\tBob Reward")
            for entry in self.history:
                round_num = entry["round"]
                alice_action = entry["actions"]["alice"]
                bob_action = entry["actions"]["bob"]
                alice_reward = entry["rewards"]["alice"]
                bob_reward = entry["rewards"]["bob"]
                output.append(f"{round_num}\t{alice_action}\t{bob_action}\t{alice_reward}\t{bob_reward}")
        
        output.append("\nTotal Rewards:")
        for agent_id in self.agent_ids:
            output.append(f"{agent_id}: {self.total_rewards[agent_id]}")
        
        return "\n".join(output)
    
    def close(self) -> None:
        """
        Perform any necessary cleanup.
        """
        pass 