import copy
import json
import random
import re
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from mllm.models.inference_backend import PolicyOutput
from mllm.markov_games.agent import Agent
from mllm.markov_games.rollout_tree import AgentActLog, ChatTurn


@dataclass
class IPDAgentState:
    """
    TOWRITE
    """

    nb_retries: int
    round_nb: int
    chat_counter: int
    chat_history: List[ChatTurn]


@dataclass
class IPDAgent(Agent):
    seed: int
    agent_id: str
    policy: Callable[[List[Dict]], str]
    intro_prompt: str  # Introduction prompt explaining the game rules
    goal_prompt: str  # Prompt explaining the agent's goal
    strategy_prompt: str  # Prompt suggesting a strategy to the agent
    max_errors: int  # Maximum number of errors allowed before default action
    allow_reasoning: bool  # Whether to allow reasoning in the response
    max_reasoning_chars: int  # Maximum number of characters for reasoning
    cooperate_string: str  # string parsed as playing cooperate by simulation
    defect_string: str  # string parsed as playing defect by simulation

    def __post_init__(self):
        self.state = IPDAgentState(
            nb_retries=0, round_nb=0, chat_counter=0, chat_history=[]
        )

    async def act(self, observation) -> Tuple[Any, AgentActLog]:
        """
        TOWRITE
        """
        action = None
        action_is_ready = False
        round_nb = observation.round_nb
        while not action_is_ready:
            # If it's the first round, we need to send the intro prompt
            if round_nb == 0 and self.state.chat_counter == 0:
                self.state.chat_history.append(
                    ChatTurn(
                        agent_id=self.agent_id,
                        role="user",
                        content=self.intro_prompt,
                        is_state_end=True,
                    )
                )

            # If new round
            if round_nb > self.state.round_nb:
                coagent_action = observation.last_coagent_move
                user_message = f"Last round, the other agent played {coagent_action}."
                self.state.chat_history.append(
                    ChatTurn(
                        agent_id=self.agent_id,
                        role="user",
                        content=user_message,
                        is_state_end=True,
                    )
                )
                self.round_nb = round_nb

            # If not new round, try to get valid action from policy
            prompt = [chat_item.dict() for chat_item in self.state.chat_history]
            policy_output: PolicyOutput = await self.policy(
                prompt=prompt, regex=f"({self.cooperate_string}|{self.defect_string})"
            )
            assert isinstance(policy_output, PolicyOutput), f"Policy output is not a PolicyOutput: {policy_output}"
            self.state.chat_history.append(
                ChatTurn(
                    agent_id=self.agent_id,
                    role="assistant",
                    content=policy_output.content,
                    reasoning_content=policy_output.reasoning_content,
                    is_state_end=False,
                )
            )

            action = policy_output
            action_is_ready = True

        self.state.nb_retries = 0  # reset retry counter
        agent_step_log = AgentActLog(
            chat_turns=self.state.chat_history[self.state.chat_counter :], info=None
        )
        self.state.chat_counter = len(self.state.chat_history)
        return action, agent_step_log

    def get_safe_copy(self):
        """
        Return a safe copy of the agent.
        """
        agent_copy = copy.copy(self)
        agent_copy.state = copy.deepcopy(self.state)
        return agent_copy

    def reset(self):
        self.state = IPDAgentState()
        raise NotImplementedError

    def render(self):
        pass

    def close(self):
        pass

    def get_agent_info(self):
        pass
