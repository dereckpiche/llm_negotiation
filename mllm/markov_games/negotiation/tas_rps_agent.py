import copy
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from mllm.markov_games.agent import Agent
from mllm.markov_games.negotiation.nego_agent import (
    Message,
    NegotiationAgent,
    NegotiationAgentState,
    Split,
)
from mllm.markov_games.negotiation.tas_rps_simulation import TrustAndSplitRPSObs
from mllm.markov_games.rollout_tree import AgentActLog, ChatTurn


class TrustAndSplitRPSAgent(NegotiationAgent):
    def __init__(
        self,
        num_message_chars: int,
        *args,
        **kwargs,
    ):
        self.num_message_chars = num_message_chars
        super().__init__(*args, **kwargs)
        self.intro_prompt = (
            "Welcome to an iterated game. You are {agent}. The other agent is {other_agent}.\n"
            "\n"
            "Setup:\n"
            "1. The game has multiple independent rounds.\n"
            "2. In each round, there are 10 coins to split between the two agents.\n"
            "3. Each agent's per-coin value for that round is determined as follows:\n"
            "   - Both agents are randomly assigned a rock, paper or scissors hands\n"
            "   - Rock has the upper hand over scissors, scissors has the upper hand over paper and paper has the upper hand over rock.\n"
            "   - The agent with the upper hand has a per-coin value of 10.\n"
            "   - The agent with the lower hand has a per-coin value of 1.\n"
            "4. You only see your own hand, but you may communicate it in messages and infer your value based on the other agent's hand.\n"
            "5. Over many rounds both agents are equally likely to have the upper and lower hand.\n"
            "\n"
            "Protocol:\n"
            "1. At the start of the round, one agent begins the conversation. The starting role alternates each round.\n"
            "2. Agents exchange a short chat ({quota_messages_per_agent_per_round} messages per round per agent) to negotiate how to split the 10 coins.\n"
            "   - Use this chat to communicate your hand so that both agents can determine their per-coin values.\n"
            "3. After the chat, both agents simultaneously propose how many coins they keep.\n"
            "4. If the total sum of proposals is less than or equal to 10, both agents receive their proposals.\n"
            "5. If the total sum of proposals exceeds 10, the coins are allocated proportionally.\n"
            "6. Your points for the round = (coins you receive) x (your per-coin value for that round). \n"
            "7. The points are accumulated across rounds.\n"
            "Your goal: {goal}\n"
        )
        self.new_round_prompt = "A new round begins\n" "Your hand is {hand}."
        self.last_round_prompt = (
            "Round summary:\n"
            "   - Your hand: {last_hand_agent}\n"
            "   - {other_agent}'s hand: {last_hand_coagent}\n"
            "   - Your value per coin: {last_value_agent}\n"
            "   - {other_agent}'s value per coin: {last_value_coagent}\n"
            "   - You proposed: {last_split_agent} coins\n"
            "   - You earned: {last_points_agent} points\n"
            "   - {other_agent} proposed: {last_split_coagent} coins\n"
            "   - {other_agent} earned: {last_points_coagent} points\n"
            "   - Round complete.\n"
        )
        self.send_split_prompt = (
            "Submit your proposal\n"
            "Respond with <coins_to_self> x </coins_to_self> where x is an integer in [0, 10]."
        )
        self.wait_for_message_prompt = "Wait for {other_agent} to send a message..."
        self.last_message_prompt = "{other_agent} said: {last_message}"
        self.send_message_prompt = f"Send your message now in <message>...</message> (<={self.num_message_chars} chars)."

    def get_message_regex(self, observation: TrustAndSplitRPSObs) -> str:
        return rf"<message>[\s\S]{{0,{self.num_message_chars}}}</message>"

    def get_split_regex(self, observation: TrustAndSplitRPSObs) -> str:
        return r"<coins_to_self> ?(10|[0-9]) ?</coins_to_self>"

    def get_split_action(
        self, policy_output: str, observation: TrustAndSplitRPSObs
    ) -> Split:
        import re as _re

        m = _re.search(r"<coins_to_self> ?(10|[0-9]) ?</coins_to_self>", policy_output)
        coins_int = int(m.group(1)) if m else int(policy_output)
        return Split(items_given_to_self={"coins": coins_int})
