import json
import random
import re
from typing import Any, Dict, List, Optional, Tuple, Union
from mllm.markov_games.agent import Agent
from collections.abc import Callable
from dataclasses import dataclass, field
from copy import deepcopy
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
    intro_prompt: str # Introduction prompt explaining the game rules
    goal_prompt: str # Prompt explaining the agent's goal
    strategy_prompt: str # Prompt suggesting a strategy to the agent
    max_errors: int # Maximum number of errors allowed before default action
    allow_reasoning: bool # Whether to allow reasoning in the response
    max_reasoning_chars: int # Maximum number of characters for reasoning
    cooperate_strings: List[str] # strings parsed as playing cooperate by simulation
    defect_strings: List[str] # strings parsed as playing defect by simulation

    def __post_init__(self):
        self.state = IPDAgentState(nb_retries=0,round_nb=0,chat_counter=0,chat_history=[])

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
                        time_step=round_nb,
                        is_state_end=True
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
                        time_step=round_nb,
                        is_state_end=True
                    )
                )
                self.round_nb = round_nb

            # If not new round, try to get valid action from policy
            policy_output = await self.policy(self.state.chat_history) # TODO: use await here!
            self.state.chat_history.append(
                ChatTurn(
                    agent_id=self.agent_id,
                    role="assistant",
                    content=policy_output,
                    time_step=round_nb,
                    is_state_end=False
                )
            )

            if policy_output in self.cooperate_strings+self.defect_strings:
                action = policy_output
                action_is_ready = True

            elif self.nb_retries < self.max_errors:
                self.state.chat_history.append(
                    ChatTurn(
                        agent_id=self.agent_id,
                        role="user",
                        content= "You have made a formatting error. Try again.",
                        time_step=round_nb,
                        is_state_end=False
                    )
                )
                self.nb_retries += 1

            else:
                action = "ERROR"
                action_is_ready = False

        self.state.nb_retries = 0  # reset retry counter
        import copy
        agent_step_log = AgentActLog(
            chat_turns = copy.deepcopy(self.state.chat_history[self.state.chat_counter:]),
            info = None
        )
        self.state.chat_counter = len(self.state.chat_history)
        return action, agent_step_log

    # def get_safe_copy(self):
    #     """

    #     """


    def reset(self):
        self.state = IPDAgentState()
        raise NotImplementedError

    def render(self):
        pass

    def close(self):
        pass


    def get_agent_info(self):
        pass
