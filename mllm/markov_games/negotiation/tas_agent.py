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
from mllm.markov_games.negotiation.nego_agent import NegotiationAgent, NegotiationAgentState


class TrustAndSplitAgent(NegotiationAgent):
    def __init__(
        self,
        num_message_chars: int,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.num_message_chars = num_message_chars
        self.state = NegotiationAgentState(
            round_nb=0, nb_messages_sent_this_round=0, chat_counter=0, chat_history=[]
        )
        self.intro_prompt = (
        "Welcome to an iterated game. You are {current_agent}. The other agent is {other_agent}.\n"
        "\n"
        "Setup:\n"
        "1. The game consists of multiple independent rounds.\n"
        "2. In each round, there are 10 coins to split between the two agents.\n"
        "3. Each round, both agents are randomly assigned rock/paper/scissors hands. The assignment is always such that one hand is winning and the other hand is losing. There are no ties.\n"
        "4. You only see your own hand.\n"
        "5. The hands are not played against each other. They only determine the value of your coin in that round:\n"
        "   - The agent with the winning hand has a coin value of 10.\n"
        "   - The agent with the losing hand has a coin value of 1.\n"
        "   - Winning hands are rock beats scissors, scissors beats paper, paper beats rock.\n"
        "   - Because assignments are random, over many rounds both agents are equally likely to have winning and losing hands.\n"
        "\n"
        "Protocol:\n"
        "1. At the start of the round, one agent begins the conversation. The starting role alternates across rounds.\n"
        "2. Agents exchange a short chat ({quota_messages} messages per round per agent) to negotiate how to split the 10 coins.\n"
        "   - Use this chat to discuss hands, strategies, and proposals.\n"
        "3. After the chat, both agents simultaneously propose how many coins they keep.\n"
        "4. If the total sum of proposals is less than or equal to 10, both agents receive their proposals.\n"
        "5. If the total sum of proposals exceeds 10, the coins are allocated proportionally.\n"
        "6. Your points for the round = (coins you receive) x (your coin value for that round). \n"
        "7. The points are accumulated across rounds.\n"
        "Your goal: {goal}\n"
        )
        self.new_round_prompt = (f"In this round, your hand is {hand}.")
        self.last_round_prompt = (f"Last round, your hand was {observation.last_hand_agent}, and {observation.other_agent}'s hand was {observation.last_hand_coagent}. Based on these hands, your value per coin was {observation.last_value_agent}, while {observation.other_agent}'s value per coin was {observation.last_value_coagent}.\nYou proposed {observation.last_split_agent} coins and earned {round(observation.last_points_agent,1)} points, while {observation.other_agent} proposed {observation.last_split_coagent} coins and earned {round(observation.last_points_coagent,1)} points.")
        self.send_split_prompt = ("Respond with <coins_to_self> x </coins_to_self> where x is an integer in [0, 10].")

    def get_message_regex(self, observation: TrustAndSplitObs) -> str:
        return rf"<message>[\s\S]{{0,{self.num_message_chars}}}</message>"
    
    def get_split_regex(self, observation: TrustAndSplitObs) -> str:
        return r"<coins_to_self>\s*(10|[0-9])\s*</coins_to_self>"
    
    def get_split_action(self, policy_output: str, observation: TrustAndSplitObs) -> Split:
        import re as _re
        m = _re.search(r"<coins_to_self>\s*(10|[0-9])\s*</coins_to_self>", policy_output)
        coins_int = int(m.group(1)) if m else int(policy_output)
        return Split(coins_given_to_self=coins_int)

  