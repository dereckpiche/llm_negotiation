from typing import Dict, Any, Tuple, List, Optional, Union
import re
import json
import random


class IPDPlayer:
    """
    Agent handler for the Iterated Prisoner's Dilemma environment.
    
    This class is responsible for:
    1. Parsing the LLM's response to extract the action (cooperate or defect)
    2. Preparing the input prompt for the LLM
    3. Tracking the agent's state throughout the game
    """
    
    def __init__(
        self,
        agent_id: str,
        policy_id: str,
        intro_prompt: str,
        goal_prompt: str = None,
        strategy_prompt: str = None,
        max_errors: int = 3,
        allow_reasoning: bool = True,
        max_reasoning_chars: int = 500,
    ):
        """
        Initialize the IPD player.
        
        Args:
            agent_id: Identifier for the agent
            policy_id: Identifier for the policy to be used
            intro_prompt: Introduction prompt explaining the game rules
            goal_prompt: Prompt explaining the agent's goal
            strategy_prompt: Prompt suggesting a strategy to the agent
            max_errors: Maximum number of errors allowed before default action
            allow_reasoning: Whether to allow reasoning in the response
            max_reasoning_chars: Maximum number of characters for reasoning
        """
        self.agent_id = agent_id
        self.policy_id = policy_id
        self.intro_prompt = intro_prompt
        self.goal_prompt = goal_prompt
        self.strategy_prompt = strategy_prompt
        self.max_errors = max_errors
        self.allow_reasoning = allow_reasoning
        self.max_reasoning_chars = max_reasoning_chars
        
        # Initialize agent state
        self.current_round = 0
        self.rounds_per_game = None
        self.history = []
        self.total_reward = 0.0
        self.error_count = 0
        self.previous_action = None
        self.opponent_id = None
        self.payoff_matrix = None
        
        # Regular expressions for parsing the LLM's response
        self.action_regex = re.compile(r'<action>(.*?)</action>', re.DOTALL)
        self.reasoning_regex = re.compile(r'<think>(.*?)</think>', re.DOTALL)
        
    def reset(self):
        """Reset the agent state."""
        self.current_round = 0
        self.history = []
        self.total_reward = 0.0
        self.error_count = 0
        self.previous_action = None
    
    def step(self, observation_from_env: Dict[str, Any], policy_output: str = None) -> Tuple[str, Dict[str, Any], str, bool, Dict[str, Any]]:
        """
        Update the agent state based on the observation and process the policy output.
        
        Args:
            observation_from_env: The observation from the environment
            policy_output: The output from the policy (LLM response)
            
        Returns:
            policy_id: The policy identifier
            policy_input: The input to the policy
            action: The action to be sent to the environment
            done: Whether the action is ready to be sent to the environment
            info: Additional information about the agent
        """
        # Update agent state from environment observation
        self.current_round = observation_from_env.get("current_round", self.current_round)
        self.rounds_per_game = observation_from_env.get("rounds_per_game", self.rounds_per_game)
        self.history = observation_from_env.get("history", self.history)
        self.total_reward = observation_from_env.get("total_reward", self.total_reward)
        self.payoff_matrix = observation_from_env.get("payoff_matrix", self.payoff_matrix)
        
        # Determine opponent's ID (for a two-player game)
        if self.opponent_id is None and self.agent_id == "alice":
            self.opponent_id = "bob"
        elif self.opponent_id is None and self.agent_id == "bob":
            self.opponent_id = "alice"
        
        # If no policy output, generate the input for the policy
        if policy_output is None:
            policy_input = self._generate_policy_input(observation_from_env)
            return self.policy_id, policy_input, None, False, {"agent_id": self.agent_id}
        
        # Parse the policy output to extract action and reasoning
        action, reasoning, error_msg = self._parse_policy_output(policy_output)
        
        # Handle parsing errors
        if error_msg:
            self.error_count += 1
            if self.error_count >= self.max_errors:
                # Default to cooperate if too many errors
                action = "C"
                error_msg = f"Too many errors ({self.error_count}). Defaulting to cooperate."
            else:
                # Request a new response
                policy_input = self._generate_policy_input(
                    observation_from_env, 
                    error_message=f"Error: {error_msg}. Please follow the action format: <action>C</action> or <action>D</action>"
                )
                return self.policy_id, policy_input, None, False, {"agent_id": self.agent_id, "error": error_msg}
        
        # Update agent state with the new action
        self.previous_action = action
        
        # Prepare info dictionary
        info = {
            "agent_id": self.agent_id,
            "current_round": self.current_round,
            "total_reward": self.total_reward,
            "action": action,
        }
        
        if reasoning:
            info["reasoning"] = reasoning
        
        return self.policy_id, None, action, True, info
    
    def _generate_policy_input(self, observation: Dict[str, Any], error_message: str = None) -> Dict[str, Any]:
        """
        Generate the input for the policy.
        
        Args:
            observation: The observation from the environment
            error_message: An error message to include in the prompt
            
        Returns:
            The input for the policy
        """
        # Build the prompt
        prompts = []
        
        # Add introduction if this is the first round
        if self.current_round == 0:
            prompts.append(self.intro_prompt)
            
            if self.goal_prompt:
                prompts.append(f"Goal: {self.goal_prompt}")
                
            if self.strategy_prompt:
                prompts.append(f"Strategy Hint: {self.strategy_prompt}")
            
            # Add explanation about how to respond
            response_format = (
                "Please respond with your action for this round by typing either 'C' for cooperate or 'D' for defect within "
                "<action></action> tags. For example: <action>C</action>"
            )
            
            if self.allow_reasoning:
                response_format += (
                    f"\nYou may also include your reasoning within <think></think> tags. "
                    f"Reasoning is limited to {self.max_reasoning_chars} characters. For example: "
                    "<think>I'll cooperate because...</think>"
                )
            
            prompts.append(response_format)
        
        # Add game state information
        prompts.append(f"\nRound {self.current_round + 1} of {self.rounds_per_game}")
        
        # Add payoff matrix information
        if self.payoff_matrix:
            prompts.append(
                f"Payoff Matrix:\n"
                f"- If both players cooperate, both receive {self.payoff_matrix['reward']} points\n"
                f"- If both players defect, both receive {self.payoff_matrix['punishment']} points\n"
                f"- If you cooperate and the other player defects, you receive {self.payoff_matrix['sucker']} points\n"
                f"- If you defect and the other player cooperates, you receive {self.payoff_matrix['temptation']} points"
            )
        
        # Add history information
        if self.history:
            history_str = "Game History:\n"
            history_str += "Round\tYour Action\tOther Action\tYour Reward\tOther Reward\n"
            
            for entry in self.history:
                round_num = entry["round"]
                your_action = entry["actions"][self.agent_id]
                other_action = entry["actions"][self.opponent_id]
                your_reward = entry["rewards"][self.agent_id]
                other_reward = entry["rewards"][self.opponent_id]
                
                history_str += f"{round_num + 1}\t{your_action}\t{other_action}\t{your_reward}\t{other_reward}\n"
            
            prompts.append(history_str)
        
        # Add total reward information
        prompts.append(f"Your total reward so far: {self.total_reward}")
        
        # Add error message if provided
        if error_message:
            prompts.append(f"\nError: {error_message}")
        
        # Add final instruction
        prompts.append("\nWhat is your action for this round? (C for cooperate, D for defect)")
        
        # Combine all prompts
        full_prompt = "\n\n".join(prompts)
        
        return {"prompt": full_prompt}
    
    def _parse_policy_output(self, policy_output: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Parse the policy output to extract the action and reasoning.
        
        Args:
            policy_output: The output from the policy
            
        Returns:
            action: The extracted action (C or D), or None if no valid action found
            reasoning: The extracted reasoning, or None if no reasoning found
            error_msg: An error message if parsing failed, or None if successful
        """
        # Extract action
        action_match = self.action_regex.search(policy_output)
        if not action_match:
            return None, None, "No action found. Please use <action>C</action> or <action>D</action> format."
        
        action = action_match.group(1).strip().upper()
        
        # Validate action
        if action not in ["C", "D"]:
            return None, None, f"Invalid action: {action}. Please use 'C' for cooperate or 'D' for defect."
        
        # Extract reasoning if present
        reasoning = None
        reasoning_match = self.reasoning_regex.search(policy_output)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
            
            # Truncate reasoning if too long
            if len(reasoning) > self.max_reasoning_chars:
                reasoning = reasoning[:self.max_reasoning_chars] + "... (truncated)"
        
        return action, reasoning, None
    
    def get_log_info(self) -> Dict[str, Any]:
        """
        Get information about the agent required to log a trajectory.
        
        Returns:
            log_info: Information about the agent for logging
        """
        return {
            "agent_id": self.agent_id,
            "policy_id": self.policy_id,
            "current_round": self.current_round,
            "total_reward": self.total_reward,
            "history": self.history,
            "previous_action": self.previous_action,
        }
    
    def render(self) -> str:
        """
        Render the current state of the agent.
        
        Returns:
            A string representation of the agent's state
        """
        output = []
        output.append(f"Agent: {self.agent_id} (using policy {self.policy_id})")
        output.append(f"Current round: {self.current_round}/{self.rounds_per_game}")
        output.append(f"Total reward: {self.total_reward}")
        
        if self.previous_action:
            output.append(f"Previous action: {self.previous_action}")
        
        return "\n".join(output)
    
    def close(self) -> None:
        """
        Perform any necessary cleanup.
        """
        pass 