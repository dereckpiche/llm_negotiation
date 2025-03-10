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
        
        # If no initial state was provided during initialization, create one
        if self.state is None:
            # Import necessary modules from deepmind_diplomacy
            from environments.diplomacy.deepmind_diplomacy import observation_utils as utils
            import numpy as np
            
            # Create a concrete implementation of the DiplomacyState protocol
            class InitialDiplomacyState:
                """Concrete implementation of the DiplomacyState protocol for a new game."""
                
                def __init__(self, random_seed=0):
                    """Initialize a new Diplomacy game state in Spring 1901."""
                    self.random_seed = random_seed
                    np.random.seed(random_seed)
                    
                    # Initialize board state to the standard starting position
                    # Create a board state with the standard starting units and supply centers
                    # The board has shape [NUM_AREAS, PROVINCE_VECTOR_LENGTH]
                    self.board_state = np.zeros(utils.OBSERVATION_BOARD_SHAPE, dtype=np.int32)
                    
                    # Set up initial supply centers
                    # Austria: Budapest, Trieste, Vienna
                    self._set_supply_center(34, 0)  # Budapest
                    self._set_supply_center(70, 0)  # Trieste
                    self._set_supply_center(71, 0)  # Vienna
                    
                    # England: Edinburgh, Liverpool, London
                    self._set_supply_center(16, 1)  # Edinburgh
                    self._set_supply_center(21, 1)  # Liverpool
                    self._set_supply_center(22, 1)  # London
                    
                    # France: Brest, Marseilles, Paris
                    self._set_supply_center(5, 2)   # Brest
                    self._set_supply_center(25, 2)  # Marseilles
                    self._set_supply_center(30, 2)  # Paris
                    
                    # Germany: Berlin, Kiel, Munich
                    self._set_supply_center(3, 3)   # Berlin
                    self._set_supply_center(20, 3)  # Kiel
                    self._set_supply_center(27, 3)  # Munich
                    
                    # Italy: Naples, Rome, Venice
                    self._set_supply_center(28, 4)  # Naples
                    self._set_supply_center(36, 4)  # Rome
                    self._set_supply_center(72, 4)  # Venice (main area of bicoastal province)
                    
                    # Russia: Moscow, Sevastopol, St Petersburg, Warsaw
                    self._set_supply_center(26, 5)  # Moscow
                    self._set_supply_center(37, 5)  # Sevastopol
                    self._set_supply_center(75, 5)  # St Petersburg (main area of bicoastal province)
                    self._set_supply_center(73, 5)  # Warsaw
                    
                    # Turkey: Ankara, Constantinople, Smyrna
                    self._set_supply_center(0, 6)   # Ankara
                    self._set_supply_center(13, 6)  # Constantinople
                    self._set_supply_center(40, 6)  # Smyrna
                    
                    # Set up initial units
                    # Austria: Army in Budapest, Army in Vienna, Fleet in Trieste
                    self._set_unit(34, utils.UnitType.ARMY.value, 0)  # Army in Budapest
                    self._set_unit(71, utils.UnitType.ARMY.value, 0)  # Army in Vienna
                    self._set_unit(70, utils.UnitType.FLEET.value, 0) # Fleet in Trieste
                    
                    # England: Fleet in Edinburgh, Fleet in London, Army in Liverpool
                    self._set_unit(16, utils.UnitType.FLEET.value, 1) # Fleet in Edinburgh
                    self._set_unit(22, utils.UnitType.FLEET.value, 1) # Fleet in London
                    self._set_unit(21, utils.UnitType.ARMY.value, 1)  # Army in Liverpool
                    
                    # France: Fleet in Brest, Army in Marseilles, Army in Paris
                    self._set_unit(5, utils.UnitType.FLEET.value, 2)  # Fleet in Brest
                    self._set_unit(25, utils.UnitType.ARMY.value, 2)  # Army in Marseilles
                    self._set_unit(30, utils.UnitType.ARMY.value, 2)  # Army in Paris
                    
                    # Germany: Fleet in Kiel, Army in Berlin, Army in Munich
                    self._set_unit(20, utils.UnitType.FLEET.value, 3) # Fleet in Kiel
                    self._set_unit(3, utils.UnitType.ARMY.value, 3)   # Army in Berlin
                    self._set_unit(27, utils.UnitType.ARMY.value, 3)  # Army in Munich
                    
                    # Italy: Fleet in Naples, Army in Rome, Army in Venice
                    self._set_unit(28, utils.UnitType.FLEET.value, 4) # Fleet in Naples
                    self._set_unit(36, utils.UnitType.ARMY.value, 4)  # Army in Rome
                    self._set_unit(72, utils.UnitType.ARMY.value, 4)  # Army in Venice
                    
                    # Russia: Fleet in Sevastopol, Fleet in St Petersburg (SC), Army in Moscow, Army in Warsaw
                    self._set_unit(37, utils.UnitType.FLEET.value, 5) # Fleet in Sevastopol
                    self._set_unit(76, utils.UnitType.FLEET.value, 5) # Fleet in St Petersburg (SC)
                    self._set_unit(26, utils.UnitType.ARMY.value, 5)  # Army in Moscow
                    self._set_unit(73, utils.UnitType.ARMY.value, 5)  # Army in Warsaw
                    
                    # Turkey: Fleet in Ankara, Army in Constantinople, Army in Smyrna
                    self._set_unit(0, utils.UnitType.FLEET.value, 6)  # Fleet in Ankara
                    self._set_unit(13, utils.UnitType.ARMY.value, 6)  # Army in Constantinople
                    self._set_unit(40, utils.UnitType.ARMY.value, 6)  # Army in Smyrna
                    
                    # Initialize game state
                    self.current_season = utils.Season.SPRING_MOVES
                    self.current_year = 1901
                    self.is_game_over = False
                    self.game_returns = np.zeros(utils.NUM_POWERS, dtype=np.float32)
                    
                    # Generate and cache legal actions for the initial state
                    self._generate_legal_actions()
                    
                    # Initialize last_actions as empty
                    self.last_actions = []
                
                def _set_supply_center(self, province_id, power_idx):
                    """Set a supply center for a specific power."""
                    # Get the area ID for the province
                    area_id, _ = utils.obs_index_start_and_num_areas(province_id)
                    # Set the supply center bit
                    self.board_state[area_id, utils.OBSERVATION_SC_POWER_START + power_idx] = 1
                
                def _set_unit(self, province_id, unit_type, power_idx):
                    """Set a unit in a specific province."""
                    # For simplicity, assume province_id is the area_id for now
                    # In a real implementation, you would use the appropriate area_id based on the province
                    area_id = province_id
                    
                    # Set unit type (ARMY or FLEET)
                    self.board_state[area_id, unit_type] = 1
                    
                    # Set unit power
                    self.board_state[area_id, utils.OBSERVATION_UNIT_POWER_START + power_idx] = 1
                
                def _generate_legal_actions(self):
                    """Generate and cache legal actions for all powers."""
                    # This would be a complex implementation using action_utils and other modules
                    # For now, we'll create a simplified placeholder
                    from environments.diplomacy.deepmind_diplomacy import action_utils
                    
                    # Create empty lists of legal actions for each power
                    self.cached_legal_actions = [[] for _ in range(utils.NUM_POWERS)]
                    
                    # For each power, generate legal moves for their units
                    for power_idx in range(utils.NUM_POWERS):
                        # Get areas with units for this power
                        areas = utils.moves_phase_areas(power_idx, self.board_state, False)
                        
                        # For each unit, generate legal actions
                        # This is a simplified placeholder
                        for area in areas:
                            # Add some placeholder legal actions
                            # In a real implementation, this would use action_utils to generate
                            # the full set of legal actions based on the board state
                            self.cached_legal_actions[power_idx].append(0)  # Hold action placeholder
                
                def is_terminal(self) -> bool:
                    """Whether the game has ended."""
                    return self.is_game_over
                
                def observation(self) -> utils.Observation:
                    """Returns the current observation."""
                    return utils.Observation(
                        board_state=self.board_state,
                        season=self.current_season,
                        year=self.current_year,
                        last_actions=self.last_actions
                    )
                
                def legal_actions(self):
                    """A list of lists of legal unit actions."""
                    return self.cached_legal_actions
                
                def returns(self):
                    """The returns of the game. All 0s if the game is in progress."""
                    return self.game_returns
                
                def step(self, actions_per_player):
                    """Steps the environment forward a full phase of Diplomacy."""
                    # This would be a complex implementation
                    # For now, we'll create a simplified progression
                    
                    # Update season
                    if self.current_season == utils.Season.SPRING_MOVES:
                        self.current_season = utils.Season.SPRING_RETREATS
                    elif self.current_season == utils.Season.SPRING_RETREATS:
                        self.current_season = utils.Season.AUTUMN_MOVES
                    elif self.current_season == utils.Season.AUTUMN_MOVES:
                        self.current_season = utils.Season.AUTUMN_RETREATS
                    elif self.current_season == utils.Season.AUTUMN_RETREATS:
                        self.current_season = utils.Season.BUILDS
                    else:  # BUILDS
                        self.current_season = utils.Season.SPRING_MOVES
                        self.current_year += 1
                    
                    # Store the actions for the observation
                    self.last_actions = []
                    for power_idx, action_list in enumerate(actions_per_player):
                        for action in action_list:
                            self.last_actions.append(action)
                    
                    # Regenerate legal actions for the new state
                    self._generate_legal_actions()
            
            # Create a new state
            self.state = InitialDiplomacyState(random_seed=self.random_seed)
            
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