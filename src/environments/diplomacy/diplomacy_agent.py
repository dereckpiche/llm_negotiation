from typing import Dict, List, Tuple, Optional, Any

class DiplomacyAgent:
    """
    Agent handler for Diplomacy, implementing the AgentState interface
    for the multi-agent negotiation standard.
    """
    
    def __init__(self, 
                 policy_id: str,
                 power_name: str,
                 use_text_interface: bool = True,
                 system_prompt: Optional[str] = None):
        """Initialize the Diplomacy agent handler.
        
        Args:
            power_name: Name of the power this agent controls
            use_text_interface: Whether to use text-based interface (vs. structured)
            system_prompt: Optional system prompt to use for the LLM
        """
        self.policy_id = policy_id
        self.power_name = power_name
        self.use_text_interface = use_text_interface
        self.system_prompt = system_prompt
        
        # Track the agent's conversation history
        self.conversation_history = []
        
        # Track if the agent has committed to an action
        self.action_ready = False
        self.current_action = None
        
        # Store the latest observation
        self.current_observation = None
        
    def step(self, observation_from_env, policy_output=None):
        """Update the agent state based on the observation and action.
        
        Args:
            observation_from_env: The observation from the environment
            policy_output: The output of the policy (LLM response)
            
        Returns:
            policy_id: The policy identifier
            policy_input: The input to the policy
            action: The official action to be sent to the environment
            done: Whether the LLM action is ready to be sent to the environment
            info: Additional information about the agent
        """
        # Store the current observation
        self.current_observation = observation_from_env
        
        # If no policy output yet, we need to generate the initial prompt
        if policy_output is None:
            # Generate the initial prompt
            policy_input = self._generate_initial_prompt()
            return  self.policy_id, policy_input, None, False, {}
        
        # Process the policy output to see if it contains a valid action
        action, is_valid = self._process_policy_output(policy_output)
        
        if is_valid:
            # If the action is valid, we're done
            self.action_ready = True
            self.current_action = action
            return  self.policy_id, None, action, True, {"valid_action": True}
        else:
            # If the action is not valid, we need to ask for clarification
            policy_input = self._generate_clarification_prompt(policy_output)
            return self.policy_id, policy_input, None, False, {"valid_action": False}
    
    def get_log_info(self):
        """Get information about the agent required to log a trajectory.
        
        Returns:
            log_info: Information about the agent required to log a trajectory.
        """
        log_info = {
            "power_name": self.power_name,
            "conversation_history": self.conversation_history,
            "current_action": self.current_action,
        }
        return log_info
    
    def render(self):
        """Render the current state of the agent."""
        # Implementation not shown for brevity
        pass
    
    def close(self):
        """Perform any necessary cleanup."""
        pass
    
    def _generate_initial_prompt(self):
        """Generate the initial prompt for the LLM.
        
        Returns:
            dict: The input to the policy (LLM)
        """
        # Start with the system prompt if provided
        system_message = self.system_prompt or self._get_default_system_prompt()
        
        # Create the user message describing the current game state
        user_message = self._create_game_state_description()
        
        # Add to conversation history
        self.conversation_history.append({"role": "system", "content": system_message})
        self.conversation_history.append({"role": "user", "content": user_message})
        
        return {
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
        }
    
    def _get_default_system_prompt(self):
        """Get the default system prompt.
        
        Returns:
            str: The default system prompt
        """
        return f"""You are playing the role of {self.power_name} in a game of Diplomacy. 
Your goal is to control as many supply centers as possible. 
You can negotiate with other players and form alliances, but remember that 
these alliances are not binding. When you need to submit orders for your units,
write them in the correct format, with each order on a new line."""
    
    def _create_game_state_description(self):
        """Create a description of the current game state.
        
        Returns:
            str: A description of the current game state
        """
        if not self.current_observation:
            return "The game is about to begin. You are playing as {self.power_name}."
        
        obs = self.current_observation
        
        # Get current season and year
        season = obs.get("current_season", "UNKNOWN")
        year = obs.get("year", 1901)
        
        # Get supply center information
        supply_centers = obs.get("supply_centers", [])
        num_supply_centers = len(supply_centers)
        
        # Get unit information
        units = obs.get("units", [])
        
        # Get possible actions in human-readable format
        possible_actions = obs.get("human_readable_actions", [])
        
        # Create the description
        description = f"""
Year: {year}, Season: {season}
You are playing as {self.power_name}.
You currently control {num_supply_centers} supply centers: {', '.join(supply_centers) or 'none'}.
Your units are: {', '.join(str(unit) for unit in units) or 'none'}.

Please provide orders for your units. Here are your possible actions:
{chr(10).join(possible_actions)}

Submit your orders, one per line, in the format like: "A MUN - BER" or "F NTH C A LON - BEL"
"""
        return description
    
    def _process_policy_output(self, policy_output):
        """Process the policy output to extract actions.
        
        Args:
            policy_output: The output from the policy (LLM)
            
        Returns:
            tuple: (action, is_valid) where action is the extracted action and is_valid is a boolean
        """
        # Add the policy output to the conversation history
        self.conversation_history.append({"role": "assistant", "content": policy_output})
        
        # Parse the policy output to extract actions
        # This would use pattern matching or similar to find action statements
        extracted_actions = self._extract_actions_from_text(policy_output)
        
        if not extracted_actions:
            return None, False
        
        # Validate the actions against the possible actions
        valid_actions = self._validate_actions(extracted_actions)
        
        if valid_actions:
            return valid_actions, True
        else:
            return None, False
    
    def _extract_actions_from_text(self, text):
        """Extract actions from text output.
        
        Args:
            text: The text to extract actions from
            
        Returns:
            list: List of extracted actions
        """
        # This would implement pattern matching to find action text
        # Placeholder implementation - would be more sophisticated in reality
        actions = []
        
        # For example, split by lines and look for action patterns
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if line and any(keyword in line for keyword in ['A ', 'F ', 'BUILD', 'REMOVE']):
                actions.append(line)
        
        return actions
    
    def _validate_actions(self, extracted_actions):
        """Validate the extracted actions against possible actions.
        
        Args:
            extracted_actions: List of actions extracted from text
            
        Returns:
            list: List of valid actions in the format expected by the environment
        """
        # This would validate the extracted actions against the possible actions
        # and convert them to the format expected by the environment
        # Placeholder implementation
        
        # In a real implementation, this would use the mila_actions.py module
        # to convert text actions to integer actions
        
        return extracted_actions
    
    def _generate_clarification_prompt(self, previous_output):
        """Generate a prompt asking for clarification on invalid actions.
        
        Args:
            previous_output: The previous output from the policy
            
        Returns:
            dict: The input to the policy (LLM)
        """
        # Create a message asking for clarification
        clarification_message = """I couldn't understand your orders. Please provide your orders in the correct format, with each order on a new line. For example:
A MUN - BER
F NTH C A LON - BEL
"""
        
        # Add to conversation history
        self.conversation_history.append({"role": "user", "content": clarification_message})
        
        # Return the updated conversation
        return {
            "messages": self.conversation_history
        }