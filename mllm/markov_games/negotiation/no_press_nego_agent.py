from typing import Any, Dict, List, Tuple

from mllm.markov_games.negotiation.nego_agent import (
    NegotiationAgent,
    NegotiationAgentState,
)
from mllm.markov_games.negotiation.nego_simulation import Split
from mllm.markov_games.negotiation.no_press_nego_simulation import NoPressObs
from mllm.markov_games.rollout_tree import AgentActLog, ChatTurn


class NoPressAgent(NegotiationAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # No communication in this variant
        self.intro_prompt = (
            "Welcome to an iterated game. You are {agent}. The other agent is {other_agent}.\n"
            "Setup:\n"
            "1. The game consists of multiple independent rounds.\n"
            "2. In each round, there are 10 coins to split between the two agents.\n"
            "3. Each round, both agents are randomly assigned a value of either 1 or 10 per coin.\n"
            "4. You can observe values of both agents.\n"
            "5. Because assignments are random, both agents are equally likely to have same expected per-coin value.\n"
            "\n"
            "Protocol:\n"
            "1. Both agents simultaneously propose how many coins they keep.\n"
            "4. If the total sum of proposals is less than or equal to 10, both agents receive their proposals.\n"
            "5. If the total sum of proposals exceeds 10, the coins are allocated proportionally.\n"
            "6. Your points for the round = (coins you receive) x (your per-coin value for that round). \n"
            "7. The points are accumulated across rounds.\n"
            "Your goal: {goal}\n"
        )
        self.new_round_prompt = "In this round, your per-coin value is {value} and {other_agent}'s per-coin value is {other_value}."
        self.last_round_prompt = "In the last round, your per-coin value was {last_value_agent} and {other_agent}'s per-coin value was {last_value_coagent}.\nYou proposed {last_split_agent} coins and earned {last_points_agent} points, while {other_agent} proposed {last_split_coagent} coins and earned {last_points_coagent} points."
        self.send_split_prompt = "Respond with <coins_to_self> X </coins_to_self> where X is the number of coins you propose for yourself, between 0 and 10 inclusive."

    def get_message_regex(self, observation: NoPressObs) -> str:
        return r"^$"  # No messages allowed

    def get_split_regex(self, observation: NoPressObs) -> str:
        return r"<coins_to_self>\s*(10|[0-9])\s*</coins_to_self>"

    def get_split_action(self, policy_output: str, observation: NoPressObs) -> Split:
        import re as _re

        m = _re.search(
            r"<coins_to_self>\s*(10|[0-9])\s*</coins_to_self>", policy_output
        )
        coins_int = int(m.group(1)) if m else int(policy_output)
        return Split(items_given_to_self={"coins": coins_int})
