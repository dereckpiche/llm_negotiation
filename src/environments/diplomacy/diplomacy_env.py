import gymnasium as gym
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

from environments.diplomacy.deepmind_diplomacy.diplomacy_state import DiplomacyState
from environments.diplomacy.deepmind_diplomacy import observation_utils
from environments.diplomacy.deepmind_diplomacy import human_readable_actions
from environments.diplomacy.deepmind_diplomacy import mila_actions


class DiplomacyEnv():
    """
    Multi-Agent Negotiation Environment for Diplomacy, adapting Deepmind's implementation
    to the MarlEnvironment standard.
    """
    def __init__(self, 
                 random_seed= 0,
                 initial_state: Optional[DiplomacyState] = None,
                 max_turns: int = 100,
                 points_per_supply_centre: bool = True,
                 forced_draw_probability: float = 0.0,
                 min_years_forced_draw: int = 35):
        """Initialize the Diplomacy environment.
        
        Args:
            initial_state: Initial DiplomacyState (optional)
            max_turns: Maximum number of turns in the game
            points_per_supply_centre: Whether to award points per supply center in case of a draw
            forced_draw_probability: Probability of forcing a draw after min_years_forced_draw
            min_years_forced_draw: Minimum years before considering a forced draw
        """
        self.random_seed = random_seed
        self.state = initial_state
        self.max_turns = max_turns
        self.points_per_supply_centre = points_per_supply_centre
        self.forced_draw_probability = forced_draw_probability
        self.min_years_forced_draw = min_years_forced_draw
        
        # Track the current turn and game history
        self.current_turn = 0
        self.game_history = []
        
        # Map player indices to power names for easier reference
        self.power_names = ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"]
        self.player_indices = {name: idx for idx, name in enumerate(self.power_names)}
        
        # Initialize pending actions
        self.pending_actions = {}
        
    def reset(self):
        """Reset the environment to an initial state and return the initial observation.
        
        Returns:
            observation (dict): A dictionary where keys are agent identifiers and values are observations.
        """
        self.current_turn = 0
        self.game_history = []
        
        # If no initial state was provided during initialization, create one (implementation not shown)
        if self.state is None:
            # This would create a new state using Deepmind's library
            # For implementation simplicity, we're assuming the state is provided
            raise ValueError("Initial state must be provided")
            
        # Get initial observations for all agents
        observation = self._get_observations()
        return observation
    
    def step(self, actions):
        """Take a step in the environment using the provided actions.

        Args:
            actions (dict): A dictionary where keys are agent identifiers and values are actions.

        Returns:
            observations (dict): A dictionary where keys are agent identifiers and values are observations.
            done (bool): Whether the episode has ended.
            info (dict): Additional information about the environment.
        """
        # Update pending actions
        for agent_id, action in actions.items():
            self.pending_actions[agent_id] = action
            
        # Check if we have actions from all expected agents
        expected_agents = self._get_active_agents()
        
        # If not all agents have provided actions, return current observations
        if not all(agent in self.pending_actions for agent in expected_agents):
            return self._get_observations(), False, {"waiting_for": [agent for agent in expected_agents if agent not in self.pending_actions]}
            
        # Convert actions from our format to Deepmind's format
        deepmind_actions = self._convert_actions_to_deepmind_format(self.pending_actions)
        
        # Step the Deepmind state
        self.state.step(deepmind_actions)
        self.current_turn += 1
        
        # Clear pending actions
        self.pending_actions = {}
        
        # Check if the game is done
        done = self.state.is_terminal() or self.current_turn >= self.max_turns
        
        # Get new observations
        observations = self._get_observations()
        
        # Create info dictionary
        info = {
            "turn": self.current_turn,
            "returns": self.state.returns() if done else None,
        }
        
        return observations, done, info
    
    def get_log_info(self):
        """Get additional information about the environment for logging.
        
        Returns:
            log_info (dict): Information about the environment required to log the game.
        """
        log_info = {
            "power_names": self.power_names,
            "game_history": self.game_history,
            "current_turn": self.current_turn,
            "current_season": self._get_current_season(),
            "supply_centers": self._get_supply_center_counts(),
        }
        return log_info
    
    def render(self):
        """Render the current state of the environment."""
        # This would implement a visualization of the current game state
        # Implementation not shown for brevity
        pass
    
    def close(self):
        """Perform any necessary cleanup."""
        pass
    
    def _get_observations(self):
        """Get observations for all active agents.
        
        Returns:
            dict: A dictionary mapping agent identifiers to observations.
        """
        raw_observation = self.state.observation()
        
        observations = {}
        for power_idx, power_name in enumerate(self.power_names):
            # Create a player-specific observation
            power_observation = {
                "board_state": raw_observation.board_state,
                "current_season": raw_observation.season,
                "player_index": power_idx,
                "possible_actions": self.state.legal_actions()[power_idx],
                "human_readable_actions": self._get_human_readable_actions(power_idx),
                "supply_centers": observation_utils.sc_provinces(power_idx, raw_observation.board_state),
                "units": self._get_power_units(power_idx, raw_observation),
                "year": raw_observation.year,
            }
            observations[power_name] = power_observation
            
        return observations
    
    def _get_active_agents(self):
        """Get the list of agents that are expected to provide actions this turn.
        
        Returns:
            list: List of agent identifiers that are active this turn.
        """
        # In Diplomacy, this would depend on the current phase and which powers still have units
        # For simplicity, we're returning all powers, but a real implementation would be more selective
        return self.power_names
    
    def _convert_actions_to_deepmind_format(self, actions_dict):
        """Convert actions from our format to Deepmind's format.
        
        Args:
            actions_dict (dict): Dictionary mapping agent IDs to actions.
            
        Returns:
            list: List of lists in Deepmind's action format.
        """
        # Initialize empty actions for all powers
        deepmind_actions = [[] for _ in range(len(self.power_names))]
        
        # Fill in the actions for powers that provided them
        for power_name, action in actions_dict.items():
            power_idx = self.player_indices[power_name]
            
            # Convert string/structured actions to Deepmind's integer format
            # This would use mila_actions.py or similar to parse text actions
            deepmind_actions[power_idx] = self._parse_action(action, power_idx)
            
        return deepmind_actions
    
    def _parse_action(self, action, power_idx):
        """Parse a text or structured action into Deepmind's action format.
        
        Args:
            action: The action in our format (text or structured)
            power_idx: Index of the power taking the action
            
        Returns:
            list: List of integer actions in Deepmind's format
        """
        # This would use mila_actions.py to parse text commands into integer actions
        # Implementation would depend on the exact format of actions in our system
        # For now, we're assuming actions are already in the correct format
        if isinstance(action, list):
            return action
        elif isinstance(action, str):
            # This would parse text commands like "A MUN - BER" into integer actions
            # using mila_actions or human_readable_actions
            # Placeholder implementation
            return []
        
        return []
    
    def _get_human_readable_actions(self, power_idx):
        """Get human-readable descriptions of possible actions for a power.
        
        Args:
            power_idx: Index of the power
            
        Returns:
            list: List of human-readable action descriptions
        """
        legal_actions = self.state.legal_actions()[power_idx]
        readable_actions = []
        
        # This would use human_readable_actions.py to convert integer actions to text
        # Placeholder implementation
        for action in legal_actions:
            readable_actions.append(str(action))  # Replace with actual conversion
            
        return readable_actions
    
    def _get_power_units(self, power_idx, observation):
        """Get the units owned by a specific power.
        
        Args:
            power_idx: Index of the power
            observation: The raw observation from the Deepmind environment
            
        Returns:
            list: List of units (location and type)
        """
        # This would extract all units owned by the specified power
        # Placeholder implementation
        units = []
        # Implementation would use observation_utils to find all units
        return units
    
    def _get_current_season(self):
        """Get the current season in the game.
        
        Returns:
            str: Current season name
        """
        season_names = ["SPRING_MOVES", "SPRING_RETREATS", "AUTUMN_MOVES", 
                         "AUTUMN_RETREATS", "BUILDS"]
        season_idx = self.state.observation().season
        return season_names[season_idx]
    
    def _get_supply_center_counts(self):
        """Get the number of supply centers owned by each power.
        
        Returns:
            dict: Dictionary mapping power names to supply center counts
        """
        observation = self.state.observation()
        counts = {}
        
        for power_idx, power_name in enumerate(self.power_names):
            supply_centers = observation_utils.sc_provinces(power_idx, observation.board_state)
            counts[power_name] = len(supply_centers)
            
        return counts 