import json
import random
import re
from typing import Any, Dict, List, Optional, Tuple, Union


class IPDAgent:
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
        Initialize the IPD agent.

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
        self.chat_history = []

    def reset(self):
        """Reset the agent state."""
        self.current_round = 0
        self.history = []
        self.total_reward = 0.0
        self.error_count = 0
        self.previous_action = None

    def step(
        self, observation_from_env: Dict[str, Any], policy_output: str = None
    ) -> Tuple[str, Dict[str, Any], str, bool, Dict[str, Any]]:
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

        # If it's the first round, we need to send the intro prompt
        if observation_from_env["round_number"] == 0:
            policy_input = self.intro_prompt
            return (
                self.policy_id,
                policy_input,
                None,
                False,
                {"agent_id": self.agent_id, "error": "First round"},
            )

        if action == "C":
            self.chat_history.append({"role": "user", "content": policy_output})
            return (
                self.policy_id,
                None,
                action,
                True,
                {"agent_id": self.agent_id, "error": None},
            )

        elif action == "D":
            self.chat_history.append({"role": "user", "content": policy_output})
            return (
                self.policy_id,
                None,
                action,
                True,
                {"agent_id": self.agent_id, "error": None},
            )
        else:
            self.chat_history.append({"role": "user", "content": policy_output})
            return (
                self.policy_id,
                None,
                None,
                False,
                {"agent_id": self.agent_id, "error": "Invalid action"},
            )

        return self.policy_id, None, action, True, info

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
