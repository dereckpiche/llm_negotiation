import copy
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from mllm.markov_games.agent import Agent
from mllm.markov_games.rollout_tree import AgentActLog, ChatTurn
from mllm.markov_games.deal_no_deal.dond_simulation import (
    DealNoDealObs,
)
from mllm.markov_games.negotiation.nego_agent import Split, NegotiationAgent, NegotiationAgentState

class DealNoDealAgent(NegotiationAgent):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.intro_prompt = (
                "You are {agent_id}. You are playing an iterated game called Deal-or-No-Deal. "
                "At each round, you and other agent will try to distribute among yourselves items of types {item_types}. "
                "You only know how much you value each item type, but not the other agent's values. "
                "You can communicate with the other agent by sending up to {quota_messages_per_agent_per_round} short messages per round. "
                "Each round, after exchanging messages, you and the other agent will submit a private proposal. "
                "A deal is accepted only if both proposals match exactly and are within stock; otherwise no deal (0 points for both at that round). "
                "The values of the items of the other agent at the previous round are revealed to you after each round. "
                "Your goal is: {goal}."
            )
        self.new_round_prompt = ("New round {round_nb}. Items: {stock}. Your values: {values}. ")
        self.last_round_prompt = ("Last round, other agent's values: {previous_values_coagent}. ")
        self.send_split_prompt = ("Respond with <split>...</split> where you propose how many items of each type you want to keep.")
        
    def get_message_regex(self, observation: DealNoDealObs) -> str:
        return r"<message>[\s\S]{0,400}</message>"
    
    def get_split_regex(self, observation: DealNoDealObs) -> str:
        parts = []
        for t in observation.item_types:
            s = int(observation.my_values.get(t, 0))
            if s <= 0:
                rng = "0"
        else:
            allowed = "|".join(str(k) for k in range(0, s + 1))
            rng = f"({allowed})"
        parts.append(fr"<{t}>{rng}</{t}>")
        items_block = "".join(parts)
        return fr"(<split>{items_block}</split>)"
    
    def get_split_action(self, policy_output: str, observation: DealNoDealObs) -> Split:
        import re as _re
        m = _re.search(r"<split>([0-9]+)</split>", policy_output)
        coins_int = int(m.group(1)) if m else int(policy_output)
        return Split(items_given_to_self=coins_int)
 


