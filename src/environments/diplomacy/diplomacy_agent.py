from typing import Dict, List, Tuple, Optional, Any
import copy

class DiplomacyAgent:
    """Agent handler for Diplomacy game that follows the MARL standard.
    
    This class is responsible for parsing LLM output into valid Diplomacy orders,
    managing the agent state, and providing information for logging.
    """
    
    def __init__(self, policy_id: str, power_name: str, random_valid_move=False):
        """Initialize the agent handler for a power in the Diplomacy game.
        
        Args:
            power_name: The name of the power this agent controls (e.g., 'FRANCE', 'ENGLAND')
            policy_id: The identifier for the policy this agent uses
            random_valid_move: If True, will select random valid moves instead of using LLM (default: False)
        """
        self.policy_id = policy_id
        self.power_name = power_name
        self.orders = []
        self.wait = True
        self.processing_state = "WAITING_FOR_ORDERS"
        self.parsed_orders = []
        self.order_status = {}
        self.message_history = []
        self.random_valid_move = random_valid_move
        
    def step(self, observation_from_env, policy_output=None):
        """Update the agent state based on the observation and LLM output.
        
        Args:
            observation_from_env: The observation from the environment
            policy_output: The output from the LLM
            
        Returns:
            policy_id: The policy identifier
            policy_input: The input to the policy
            action: The official action to be sent to the environment
            done: Whether the LLM action is ready to be sent to the environment
            info: Additional information about the agent
        """
        info = {}
        
        # If random_valid_move is enabled, select random valid moves
        if self.random_valid_move:
            valid_orders = self._select_random_valid_moves(observation_from_env)
            self.orders = valid_orders
            self.wait = False
            action = {
                "orders": valid_orders,
                "wait": False
            }
            return self.policy_id, {}, action, True, info
        
        # If no policy output, this is the initial step - prepare prompt
        if policy_output is None:
            # Create initial prompt for the LLM
            phase = observation_from_env.get('phase', '')
            units = observation_from_env.get('units', {}).get(self.power_name, [])
            centers = observation_from_env.get('centers', {}).get(self.power_name, [])
            orderable_locations = observation_from_env.get('orderable_locations', {})
            
            prompt = self._create_prompt(phase, units, centers, orderable_locations)
            
            return self.policy_id, {"prompt": prompt}, None, False, info
            
        # Process the LLM output to extract orders
        success, parsed_orders = self._parse_llm_output(policy_output)
        self.parsed_orders = parsed_orders
        
        if not success:
            # Need more information from LLM
            clarification_prompt = self._create_clarification_prompt(policy_output, parsed_orders)
            return self.policy_id, {"prompt": clarification_prompt}, None, False, info
        
        # Validate if the orders are valid for the current phase
        valid_orders = self._validate_orders(parsed_orders, observation_from_env)
        
        if valid_orders:
            # Orders are valid, prepare action for environment
            self.orders = valid_orders
            self.wait = False
            action = {
                "orders": valid_orders,
                "wait": False
            }
            return self.policy_id, {}, action, True, info
        else:
            # Orders are invalid, ask for new ones
            error_prompt = self._create_error_prompt(parsed_orders, observation_from_env)
            return self.policy_id, {"prompt": error_prompt}, None, False, info
    
    def _create_prompt(self, phase, units, centers, orderable_locations):
        """Create the initial prompt for the LLM.
        
        Args:
            phase: The current game phase
            units: List of units controlled by this power
            centers: List of supply centers controlled by this power
            orderable_locations: List of locations where orders can be issued
            
        Returns:
            A prompt string for the LLM
        """
        prompt = f"You are playing as {self.power_name} in Diplomacy. The current phase is {phase}.\n\n"
        prompt += f"Your units: {', '.join(units)}\n"
        prompt += f"Your supply centers: {', '.join(centers)}\n"
        prompt += f"Locations you can order: {', '.join(orderable_locations)}\n\n"
        
        if phase.endswith('M'):  # Movement phase
            prompt += "Please provide orders for your units in the form:\n"
            prompt += "- A LON H (hold)\n"
            prompt += "- F NTH - NWY (move)\n"
            prompt += "- A WAL S F LON (support)\n"
            prompt += "- F NWG C A NWY - EDI (convoy)\n"
        elif phase.endswith('R'):  # Retreat phase
            prompt += "Please provide retreat orders for your dislodged units:\n"
            prompt += "- A PAR R MAR (retreat to MAR)\n"
            prompt += "- A PAR D (disband)\n"
        elif phase.endswith('A'):  # Adjustment phase
            if len(units) < len(centers):
                prompt += "You can build units. Please provide build orders:\n"
                prompt += "- A PAR B (build army in PAR)\n"
                prompt += "- F BRE B (build fleet in BRE)\n"
                prompt += "- WAIVE (waive a build)\n"
            elif len(units) > len(centers):
                prompt += "You must remove units. Please provide disbandment orders:\n"
                prompt += "- A PAR D (disband army in PAR)\n"
                prompt += "- F BRE D (disband fleet in BRE)\n"
        
        prompt += "\nProvide your orders as a list, one per line."
        return prompt
    
    def _parse_llm_output(self, llm_output):
        """Parse the LLM output to extract orders.
        
        Args:
            llm_output: The raw output from the LLM
            
        Returns:
            success: Whether parsing was successful
            parsed_orders: List of parsed orders
        """
        # Simple parsing for now - extract lines that look like orders
        lines = llm_output.strip().split('\n')
        orders = []
        
        for line in lines:
            # Remove list markers, hyphens, etc.
            line = line.strip('- *•').strip()
            
            # Skip empty lines and lines that don't look like orders
            if not line or line.startswith('I ') or line.startswith('Let\'s'):
                continue
                
            # Check if it looks like a Diplomacy order
            if (' H' in line or ' -' in line or ' S ' in line or ' C ' in line or 
                ' R ' in line or ' D' in line or ' B' in line or line == 'WAIVE'):
                orders.append(line)
        
        return len(orders) > 0, orders
    
    def _validate_orders(self, orders, observation):
        """Validate if the orders are valid for the current phase.
        
        Args:
            orders: List of orders to validate
            observation: Current observation from the environment
            
        Returns:
            List of valid orders or None if invalid
        """
        # For simplicity, we'll assume all parsed orders are valid
        # In a real implementation, we would use the game's validation logic
        return orders
    
    def _create_clarification_prompt(self, previous_output, parsed_orders):
        """Create a prompt asking for clarification when orders couldn't be parsed.
        
        Args:
            previous_output: The previous LLM output
            parsed_orders: Any orders that were successfully parsed
            
        Returns:
            A prompt string for the LLM
        """
        prompt = f"I couldn't fully understand your orders for {self.power_name}. "
        
        if parsed_orders:
            prompt += f"I understood these orders:\n"
            for order in parsed_orders:
                prompt += f"- {order}\n"
                
        prompt += "\nPlease provide clear, valid Diplomacy orders in the format:\n"
        prompt += "- A LON H\n- F NTH - NWY\n- etc.\n"
        return prompt
    
    def _create_error_prompt(self, invalid_orders, observation):
        """Create a prompt when orders are invalid.
        
        Args:
            invalid_orders: The invalid orders
            observation: Current observation from the environment
            
        Returns:
            A prompt string for the LLM
        """
        prompt = f"The following orders for {self.power_name} are invalid:\n"
        for order in invalid_orders:
            prompt += f"- {order}\n"
            
        prompt += "\nPlease provide valid orders for your units."
        return prompt
    
    def get_log_info(self):
        """Get information about the agent required to log a trajectory.
        
        Returns:
            log_info: Information about the agent required to log a trajectory.
        """
        return {
            "power_name": self.power_name,
            "orders": self.orders,
            "wait": self.wait,
            "parsing_state": self.processing_state,
            "message_history": self.message_history
        }
    
    def render(self):
        """Render the current state of the agent."""
        print(f"Power: {self.power_name}")
        print(f"Orders: {self.orders}")
        print(f"Wait: {self.wait}")
    
    def close(self):
        """Perform any necessary cleanup."""
        pass

    def _select_random_valid_moves(self, observation):
        """Select random valid moves for all units.
        
        Args:
            observation: Current observation from the environment
            
        Returns:
            List of valid orders
        """
        import random
        
        possible_orders = observation.get('possible_orders', {})
        valid_orders = []
        
        # For each location with possible orders, select one randomly
        for location, orders in possible_orders.items():
            if orders:  # If there are any possible orders for this location
                valid_orders.append(random.choice(orders))
        
        return valid_orders