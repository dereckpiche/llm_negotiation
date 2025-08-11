import copy
import json
import random
import re
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from mllm.markov_games.agent import Agent
from mllm.markov_games.rollout_tree import AgentActLog, ChatTurn
from mllm.markov_games.trust_and_split.tas_simulation import (
    Message,
    Split,
    TrustAndSplitObs,
)


@dataclass
class TrustAndSplitAgentState:
    round_nb: int
    nb_messages_sent_this_round: int
    chat_counter: int
    chat_history: List[ChatTurn]


intro_prompt = """
Welcome, you are participating in an iterated 2-agents game.
You are playing as the agent named {agent_name}.
The other agent with whom you are playing is named {coagent_name}.
In this game, two agents bargain over how to divide 10 coins over multiple rounds.
At each round, coins are valued differently by each agent.
At the end of each round, each agent privatly sends the number of coins they want to keep for themselves.
The coins are then attributed based on the splits.
If the sum of the splits is not 10, the coins are distributed proportionally to the splits.
For example, if both players propose to take all of the 10 coins, both agents get 5 coins.

Each agent then receives number of coins obtained * their value as a reward.

Messages are sent as <message>your-message-here</message>.
Splits are sent as <coins_to_self>x</coins_to_self>, where `x` is the number of coins you give to yourself.
After each round, the values of the other agent from the previous round will be reveiled.
"""
previous_round_ended_str = """
The previous round has ended.
In the last, round, {coagent_name} valued coins at {last_round_coagent_values}.
You obtained a reward of {last_round_agent_reward}.
{coagent_name} obtained a reward of {last_round_coagent_reward}.
"""
new_round_prompt = """
A new round has started. For this round, your values for each coin is of {self.value}.
"""
other_agent_split_str = """
You must now send your split.
"""
other_agent_sent_message_str = """
The other agent sent the following message: {last_message}.
"""


class TrustAndSplitAgent(Agent):
    def __init__(
        self,
        seed: int,
        agent_id: str,
        policy: Callable[[List[Dict]], str],
        nb_messages_per_round: int,
    ):
        self.seed = seed
        self.agent_id = agent_id
        self.policy = policy
        self.nb_messages_per_round = nb_messages_per_round
        self.state = TrustAndSplitAgentState(
            round_nb=0, chat_counter=0, chat_history=[]
        )

    async def act(self, observation: TrustAndSplitObs) -> Tuple[Any, AgentActLog]:
        """
        TOWRITE
        """
        action = None
        round_nb = observation.round_nb
        previous_round_ended = round_nb > self.state.round_nb

        #################################################
        # Game Starting
        #################################################
        if round_nb == 0 and self.state.chat_counter == 0:
            self.state.chat_history.append(
                ChatTurn(
                    agent_id=self.agent_id,
                    role="user",
                    content=intro_prompt.format(
                        agent_name=self.agent_id, coagent_name=self.coagent_id
                    ),
                    is_state_end=True,
                )
            )

        #################################################
        # Round Starting
        #################################################
        if round_nb > self.state.round_nb:
            self.state.nb_messages_sent_this_round = 0
            prompt = ""
            if previous_round_ended:
                prompt = previous_round_ended_str.format(
                    last_round_agent_finalization=self.state.last_round_agent_finalization,
                    last_round_agent_values=self.state.last_round_agent_values,
                    last_round_coagent_finalization=self.state.last_round_coagent_finalization,
                    last_round_coagent_values=self.state.last_round_coagent_values,
                    last_round_agent_reward=self.state.last_round_agent_reward,
                    last_round_coagent_reward=self.state.last_round_coagent_reward,
                )
            prompt += new_round_prompt.format(self.value)
            self.state.chat_history.append(
                ChatTurn(
                    agent_id=self.agent_id,
                    role="user",
                    content=prompt,
                    is_state_end=True,
                )
            )
            self.round_nb = round_nb

            # If not new round, try to get valid action from policy
            prompt = [chat_item.dict() for chat_item in self.state.chat_history]
            policy_output = await self.policy(
                prompt=prompt, regex=f"({self.cooperate_string}|{self.defect_string})"
            )
            self.state.chat_history.append(
                ChatTurn(
                    agent_id=self.agent_id,
                    role="assistant",
                    content=policy_output,
                    is_state_end=False,
                )
            )

            action = policy_output

        #################################################
        # Messages
        #################################################
        if self.state.nb_messages_sent_this_round < self.nb_messages_per_round:
            prompt = ""
            prompt += other_agent_sent_message_str.format(
                last_message=self.state.last_message
            )
            self.state.chat_history.append(
                ChatTurn(
                    agent_id=self.agent_id,
                    role="user",
                    content=prompt,
                    is_state_end=True,
                )
            )
            return_regex = r"<message>(.*){0,400}</message>"  # Any character (number of chars between 0 and 400)
            message = await self.policy(prompt=prompt, regex=return_regex)
            action = Message(message=message)
            self.state.nb_messages_sent_this_round += 1

        #################################################
        # Split
        #################################################
        if self.state.nb_messages_sent_this_round == self.nb_messages_per_round:
            prompt = ""
            prompt += other_agent_sent_message_str.format(
                last_message=self.state.last_message
            )
            prompt += other_agent_split_str.format(last_message=self.state.last_message)
            self.state.chat_history.append(
                ChatTurn(
                    agent_id=self.agent_id,
                    role="user",
                    content=prompt,
                    is_state_end=True,
                )
            )
            return_regex = r"<coins_to_self>([0-9]+)</coins_to_self>"
            item_to_self = await self.policy(prompt=prompt, regex=return_regex)
            self.state.chat_history.append(
                ChatTurn(
                    agent_id=self.agent_id,
                    role="assistant",
                    content=item_to_self,
                    is_state_end=False,
                )
            )
            item_to_self = int(item_to_self)  # TODO (get int inside split string)
            action = Split(coins_given_to_self=item_to_self)

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
        self.state = TrustAndSplitAgentState(
            round_nb=0, chat_counter=0, chat_history=[]
        )
