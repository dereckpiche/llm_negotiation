import json
import random
import re
from typing import Any, Dict, List, Optional, Tuple, Union
from mllm.markov_games.agent import Agent
from collections.abc import Callable
from dataclasses import dataclass
from copy import deepcopy

@dataclass
class IPDAgentState:
    """
    (...)
    """
    nb_retries: int = 0
    round_nb: int = 0
    chat_counter: int = 0
    chat_history: List[Dict] = []

@dataclass
class IPDAgent(Agent):

    agent_id: str
    policy: Callable[[List[Dict]], str]
    intro_prompt: str # Introduction prompt explaining the game rules
    goal_prompt: str # Prompt explaining the agent's goal
    strategy_prompt: str # Prompt suggesting a strategy to the agent
    max_errors: int # Maximum number of errors allowed before default action
    allow_reasoning: bool # Whether to allow reasoning in the response
    max_reasoning_chars: int # Maximum number of characters for reasoning
    cooperate_strings: List[str] # strings parsed as playing cooperate by simulation
    defect_strings: List[str] # strings parsed as playing defect by simulation
    state = IPDAgentState()

    async def act(self, observation):
        """

        """
        action = None
        action_is_ready = False
        round_nb = observation.round_nb

        while not action_is_ready:

            # If it's the first round, we need to send the intro prompt
            if round_nb == 0 and self.state.chat_counter == 0:
                self.state.chat_history.append(
                    {
                        "role": "user",
                        "content": self.intro_prompt,
                        "round_nb": round_nb,
                    }
                )

            # If new round
            if round_nb > self.state.round_nb:
                coagent_action = observation.last_move_coagent
                user_message = f"Last round, the other agent played {coagent_action}."
                self.state.chat_history.append(
                    {
                        "role": "user",
                        "content": user_message,
                        "round_nb": round_nb,
                    }
                )
                self.round_nb = round_nb

            # If not new round, try to get valid action from policy
            policy_output = self.policy(self.state.chat_history)
            self.state.chat_history.append(
                {
                    "role": "assistant",
                    "content": policy_output,
                    "round_nb": round_nb,
                }
            )

            if policy_output in self.cooperate_strings+self.defect_strings:
                action = policy_output
                action_is_ready = True

            elif self.nb_retries < self.max_errors:
                self.state.chat_history.append(
                    {
                        "role": "user",
                        "content": "You have made a formatting error. Try again.",
                        "is_error": True,
                        "round_nb": round_nb,
                    }
                )
                self.nb_retries += 1

            else:
                action = "ERROR"
                action_is_ready = False

        self.state.nb_retries = 0  # reset retry counter

        info = deepcopy(self.state.chat_history[self.state.chat_counter:])

        self.state.chat_counter = len(self.state.chat_history)

        return action, info

    def reset(self):
        self.state = IPDAgentState()
        raise NotImplementedError

    def render(self):
        pass

    def close(self):
        pass

    def seed(self):
        pass

    def get_agent_info(self):
        pass
