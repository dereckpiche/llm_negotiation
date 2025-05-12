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
        self.agent_name = agent_id  # TODO: fix this (for backward comp.)
        self.policy_id = policy_id
        self.intro_prompt = intro_prompt
        self.goal_prompt = goal_prompt
        self.strategy_prompt = strategy_prompt
        self.max_errors = max_errors
        self.allow_reasoning = allow_reasoning
        self.max_reasoning_chars = max_reasoning_chars
        self.chat_history = []
        self.round_nb = 0

    def reset(self):
        """Reset the agent state."""
        self.current_round = 0
        self.history = []
        self.total_reward = 0.0
        self.error_count = 0
        self.previous_action = None

    def step(
        self, observation_from_env: Any, policy_output: str = None
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

        round_nb = observation_from_env.round_nb

        # If it's the first round, we need to send the intro prompt
        if round_nb == 0 and len(self.chat_history) == 0:
            self.chat_history.append(
                {
                    "role": "user",
                    "content": self.intro_prompt,
                    "round_number": round_nb,
                }
            )
            return (
                self.policy_id,
                self.chat_history,
                None,
                False,
                None,
            )

        # If new round
        if round_nb > self.round_nb:
            other_player_id = (
                observation_from_env.agent_ids[1]
                if self.agent_id == observation_from_env.agent_ids[0]
                else observation_from_env.agent_ids[0]
            )
            other_player_action = observation_from_env.actions[-1][other_player_id]

            if other_player_action == "C":
                user_message = f"Last round, the other player cooperated."
            elif other_player_action == "D":
                user_message = f"Last round, the other player defected."
            else:
                user_message = (
                    f"Last round, the other player did not play a valid action."
                )

            self.chat_history.append(
                {
                    "role": "user",
                    "content": user_message,
                    "round_number": round_nb,
                }
            )
            self.round_nb = round_nb
            return (self.policy_id, self.chat_history, None, False, None)

        # If not new round we take action
        if policy_output in ["<Cooperate>", "<A>"]:
            action = "C"
        elif policy_output in ["<Defect>", "<B>"]:
            action = "D"
        else:
            action = "ERROR"
        self.chat_history.append(
            {
                "role": "assistant",
                "content": policy_output,
                "round_number": round_nb,
            }
        )
        return (
            None,
            None,
            action,
            True,
            None,
        )

    def get_log_info(self) -> Dict[str, Any]:
        """
        Get information about the agent required to log a trajectory.

        Returns:
            log_info: Information about the agent for logging
        """
        log_info = {
            "agent_id": self.agent_id,
            "policy_id": self.policy_id,
            "chat_history": self.chat_history,
        }
        return log_info

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
