import copy
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from mllm.markov_games.agent import Agent
from mllm.markov_games.rollout_tree import AgentActLog, ChatTurn
from mllm.markov_games.trust_and_split.tas_simulation import (
    Message,
    Split,
    TrustAndSplitObs,
)
from mllm.markov_games.negotiation.nego_agent import NegotiationAgent


@dataclass
class TrustAndSplitAgentState:
    round_nb: int
    nb_messages_sent_this_round: int
    chat_counter: int
    chat_history: List[ChatTurn]


INTRO_PROMPT = (
    "Welcome to an iterated game. You are {agent_name}.\n"
    "\n"
    "Setup:\n"
    "1. Each round, both agents receive random rock/paper/scissors hands.\n"
    "2. You see only your own hand. The winning hands are rock > scissors, scissors > paper, paper > rock.\n"
    "3. These hands are not meant to be played, but to know the value assigned to you for each coin.\n"
    "4. The agent with the winning hand has a value of 10 per coin, while the agent with the losing hand has a value of 1 per coin.\n"
    "5. There are 10 coins to split.\n"
    "\n"
    "Protocol:\n"
    "1. Short chat ({quota_messages} messages per round per agent), then both propose how many coins they keep.\n"
    "2. You are strongly encouraged to use your messages to discuss hands. \n"
    "3. The points you get are the number of coins you keep times your value (either 10 or 1). \n"
    "4. If the sum of the proposals exceeds 10, the allocation is proportional.\n"
    "\n"
    "Your goal: {goal}"
)


class TrustAndSplitAgent(NegotiationAgent):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.state = TrustAndSplitAgentState(
            round_nb=0, nb_messages_sent_this_round=0, chat_counter=0, chat_history=[]
        )
        self.intro_prompt = (
            "Welcome to an iterated game. You are {agent_name}.\n"
            "\n"
            "Setup:\n"
            "1. Each round, both agents receive random rock/paper/scissors hands.\n"
            "2. You see only your own hand. The winning hands are rock > scissors, scissors > paper, paper > rock.\n"
            "3. These hands are not meant to be played, but to know the value assigned to you for each coin.\n"
            "4. The agent with the winning hand has a value of 10 per coin, while the agent with the losing hand has a value of 1 per coin.\n"
            "5. There are 10 coins to split.\n"
            "\n"
            "Protocol:\n"
            "1. Short chat ({quota_messages} messages per round per agent), then both propose how many coins they keep.\n"
            "2. You are strongly encouraged to use your messages to discuss hands. \n"
            "3. The points you get are the number of coins you keep times your value (either 10 or 1). \n"
            "4. If the sum of the proposals exceeds 10, the allocation is proportional.\n"
            "\n"
            "Your goal: {goal}"
        ).format(agent_name=self.agent_id, goal=self.goal, quota_messages=self.quota_messages_per_agent_per_round)

    def get_message_regex(self, observation: TrustAndSplitObs) -> str:
        return r"<message>[\s\S]{0,400}</message>"
    
    def get_split_regex(self, observation: TrustAndSplitObs) -> str:
        return r"<coins_to_self>(10|[0-9])</coins_to_self>"
    
    def get_split_action(self, policy_output: str, observation: TrustAndSplitObs) -> Split:
        import re as _re
        m = _re.search(r"<coins_to_self>([0-9]+)</coins_to_self>", policy_output)
        coins_int = int(m.group(1)) if m else int(policy_output)
        return Split(coins_given_to_self=coins_int)

