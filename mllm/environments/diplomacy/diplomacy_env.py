from typing import Dict, List, Tuple, Optional, Any
from diplomacy import Game
import random

class DiplomacyEnv:
    """Multi-Agent Reinforcement Learning environment for Diplomacy.
    
    This class wraps the Diplomacy game engine to provide an interface
    compliant with the MARL standard.
    """
    
    def __init__(self, random_seed=None, map_name="standard", game_id=None, rules=None, max_steps=50):
        """Initialize the Diplomacy environment.
        
        Args:
            map_name: The name of the map to use (default: "standard")
            game_id: Optional game ID
            rules: Optional rules to apply to the game
            max_steps: Maximum number of steps before forcing game end (default: 10)
        """
        self.random_seed = random_seed
        self.map_name = map_name
        self.game_id = game_id
        self.rules = rules or []
        self.game = None
        self.active_powers = []
        self.render_mode = None
        self.max_steps = max_steps
        self.current_steps = 0
    
    def reset(self):
        """Reset the environment to an initial state and return the initial observation.
        
        Returns:
            observation: A dictionary where keys are agent identifiers and values are observations.
        """
        # Initialize a new game
        self.game = Game(game_id=self.game_id, map_name=self.map_name)
        
        # Apply rules
        for rule in self.rules:
            self.game.add_rule(rule)
        
        # Determine active powers (not eliminated)
        self.active_powers = [name for name, power in self.game.powers.items() 
                             if not power.is_eliminated()]
        
        # Reset step counter
        self.current_steps = 0
        
        # Create initial observations for all powers
        observations = {}
        for power_name in self.active_powers:
            observations[power_name] = self._create_observation(power_name)
        
        return observations
    
    def step(self, actions):
        """Take a step in the environment using the provided actions.
        
        Args:
            actions: A dictionary where keys are agent identifiers and values are actions.
            
        Returns:
            observations: A dictionary where keys are agent identifiers and values are observations.
            done: Whether the episode has ended.
            info: Additional information about the environment.
        """
        print(f"stepping {self.current_steps}")
        self.current_steps += 1
        # Apply actions (orders) for each power
        for power_name, action in actions.items():
            if power_name in self.active_powers:
                orders = action.get("orders", [])
                wait = action.get("wait", True)
                
                # Set orders for the power
                if orders:
                    self.game.set_orders(power_name, orders)
                
                # Set wait flag
                self.game.set_wait(power_name, wait)
        
        # Check if all active powers are ready to proceed
        if self.game.does_not_wait():
            # Process the current phase
            self.game.process()
            
            
            # Update active powers list after processing
            self.active_powers = [name for name, power in self.game.powers.items() 
                                 if not power.is_eliminated()]
        
        # Create observations for all active powers
        observations = {}
        for power_name in self.active_powers:
            observations[power_name] = self._create_observation(power_name)
        
        # Check if the game is done (either naturally or due to max steps)
        done = self.game.is_game_done or self.current_steps >= self.max_steps
        
        # Create info dict
        info = {
            "phase": self.game.get_current_phase(),
            "active_powers": self.active_powers,
            "centers": self.game.get_centers(),
            "units": self.game.get_units(),
            "current_steps": self.current_steps,
            "max_steps_reached": self.current_steps >= self.max_steps
        }
        
        return observations, done, info
    
    def _create_observation(self, power_name):
        """Create observation for a specific power.
        
        Args:
            power_name: The name of the power
            
        Returns:
            An observation dictionary
        """
        observation = {
            "phase": self.game.get_current_phase(),
            "units": self.game.get_units(),
            "centers": self.game.get_centers(),
            "orderable_locations": self.game.get_orderable_locations(power_name),
            "order_status": self.game.get_order_status(power_name),
            "possible_orders": self._get_possible_orders_for_power(power_name)
        }
        return observation
    
    def _get_possible_orders_for_power(self, power_name):
        """Get all possible orders for a power's units.
        
        Args:
            power_name: The name of the power
            
        Returns:
            A dictionary mapping units to their possible orders
        """
        all_possible_orders = self.game.get_all_possible_orders()
        
        # Filter for only the locations where this power has units
        power_units = self.game.get_units(power_name)
        power_unit_locations = [unit[2:] for unit in power_units]
        
        # For retreat phases, include retreating units
        if self.game.phase_type == 'R':
            power = self.game.get_power(power_name)
            power_unit_locations.extend([unit[2:] for unit in power.retreats])
        
        # For adjustment phases, include buildable locations
        elif self.game.phase_type == 'A':
            power = self.game.get_power(power_name)
            # If we have more centers than units, we can build
            if len(power.centers) > len(power.units):
                buildable_sites = self.game._build_sites(power)
                power_unit_locations.extend(buildable_sites)
            # If we have more units than centers, we need to remove
            elif len(power.units) > len(power.centers):
                # All units are candidates for removal
                pass
        
        # Filter the possible orders to only those for this power's units/locations
        power_possible_orders = {}
        for loc, orders in all_possible_orders.items():
            if loc[:3] in power_unit_locations:
                power_possible_orders[loc] = orders
        
        return power_possible_orders
    
    def get_log_info(self):
        """Get additional information about the environment for logging.
        
        Returns:
            log_info: Information about the environment required to log the game.
        """
        if not self.game:
            return {}
            
        return {
            "game_id": self.game.game_id,
            "phase": self.game.get_current_phase(),
            "map_name": self.game.map_name,
            "centers": self.game.get_centers(),
            "units": self.game.get_units(),
            "powers": {name: {
                "units": power.units,
                "centers": power.centers,
                "is_eliminated": power.is_eliminated(),
                "order_status": self.game.get_order_status(name)
            } for name, power in self.game.powers.items()},
            "orders": self.game.get_orders(),
            "active_powers": self.active_powers,
            "is_game_done": self.game.is_game_done,
            "outcome": self.game.outcome if self.game.is_game_done else None
        }
    
    def render(self, mode='human'):
        """Render the current state of the environment.
        
        Args:
            mode: The rendering mode ('human', 'svg', etc.)
        
        Returns:
            The rendered image if applicable
        """
        self.render_mode = mode
        if self.game:
            if mode == 'human':
                # Just print basic game state
                print(f"Game: {self.game.game_id}")
                print(f"Phase: {self.game.get_current_phase()}")
                print(f"Active Powers: {self.active_powers}")
                print("Supply Centers:")
                for power_name, centers in self.game.get_centers().items():
                    print(f"  {power_name}: {centers}")
                print("Units:")
                for power_name, units in self.game.get_units().items():
                    print(f"  {power_name}: {units}")
                return None
            elif mode == 'svg':
                # Return SVG representation
                return self.game.render(output_format='svg')
        return None
    
    def close(self):
        """Perform any necessary cleanup."""
        self.game = None