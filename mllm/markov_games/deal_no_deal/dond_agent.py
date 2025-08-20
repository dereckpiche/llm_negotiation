import copy
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from mllm.markov_games.agent import Agent
from mllm.markov_games.rollout_tree import AgentActLog, ChatTurn
from mllm.markov_games.deal_no_deal.dond_simulation import (
    DealNoDealObs,
    Message,
    Proposal,
)


@dataclass
class DealNoDealAgentState:
    round_nb: int
    nb_messages_sent_this_round: int
    chat_counter: int
    chat_history: List[ChatTurn]


def proposal_regex(item_types: List[str], stock: Dict[str, int]) -> str:
    parts = []
    for t in item_types:
        s = int(stock.get(t, 0))
        if s <= 0:
            rng = "0"
        else:
            allowed = "|".join(str(k) for k in range(0, s + 1))
            rng = f"({allowed})"
        parts.append(fr"<{t}>{rng}</{t}>")
    items_block = "".join(parts)
    return fr"(<propose>{items_block}</propose>|<no_deal/>)"


class DealNoDealAgent(Agent):
    def __init__(
        self,
        seed: int,
        agent_id: str,
        policy: Callable[[List[Dict]], str],
        nb_messages_per_round: int,
        goal: str,
    ):
        self.seed = seed
        self.agent_id = agent_id
        self.policy = policy
        self.goal = goal
        self.state = DealNoDealAgentState(
            round_nb=0, nb_messages_sent_this_round=0, chat_counter=0, chat_history=[]
        )

    async def act(self, observation: DealNoDealObs) -> Tuple[Any, AgentActLog]:
        is_our_turn = observation.current_agent == self.agent_id
        action: Any = None
        round_nb = observation.round_nb

        prompt_parts: List[str] = []

        if round_nb == 0 and self.state.chat_counter == 0:
            intro = (
                f"You are {self.agent_id}. You are playing an iterated game called Deal-or-No-Deal. "
                f"At each round, you and other agent will try to distribute among yourselves items of types {observation.item_types}. "
                f"You only know how much you value each item type, but not the other agent's values. "
                f"You can communicate with the other agent by sending up to {observation.quota_messages_per_agent_per_round} short messages per round. "
                f"Each round, after exchanging messages, you and the other agent will submit a private proposal. "
                f"A deal is accepted only if both proposals match exactly and are within stock; otherwise no deal (0 points for both at that round). "
                f"The values of the items of the other agent at the previous round are revealed to you after each round. "
                f"Your goal is: {self.goal}."
            )
            prompt_parts.append(intro)

        is_new_round = (round_nb > self.state.round_nb) or (self.state.round_nb == 0)
        if is_new_round:
            self.state.nb_messages_sent_this_round = 0
            prompt_parts.append(
                f"Round {round_nb}. Items: {observation.stock}. Your values: {observation.my_values}."
                f" Values of other agent at previous round: {observation.previous_values_coagent}."
            )
            self.state.round_nb = round_nb

        if not is_our_turn and not observation.proposal_phase:
            prompt_parts.append("Wait for the other agent to send a message...")

        if is_our_turn and not is_new_round and not observation.proposal_phase:
            prompt_parts.append(f"Other agent said: {observation.last_message}")

        must_send_message = (
            not observation.proposal_phase
            and is_our_turn
        )
        if must_send_message:
            prompt_parts.append(
                "Send <message>...</message> (<=400 chars)."
            )

        must_send_proposal = not must_send_message and observation.proposal_phase
        if must_send_proposal:
            prompt_parts.append(
                "Submit either <no_deal/> or <propose>per-item integers within stock and summing to total stock</propose>."
            )

        user_prompt = "\n".join(prompt_parts)
        self.state.chat_history.append(
            ChatTurn(
                agent_id=self.agent_id,
                role="user",
                content=user_prompt,
                is_state_end=True,
            )
        )

        if must_send_message:
            return_regex = r"<message>[\s\S]{0,400}</message>"
            policy_output = await self.policy(
                prompt=[c.dict() for c in self.state.chat_history],
                regex=return_regex,
            )
            self.state.chat_history.append(
                ChatTurn(
                    agent_id=self.agent_id,
                    role="assistant",
                    content=policy_output,
                    is_state_end=False,
                )
            )
            action = Message(message=policy_output)
            self.state.nb_messages_sent_this_round += 1
            
        elif must_send_proposal:
            return_regex = proposal_regex(observation.item_types, observation.stock)
            policy_output = await self.policy(
                prompt=[c.dict() for c in self.state.chat_history],
                regex=return_regex,
            )
            self.state.chat_history.append(
                ChatTurn(
                    agent_id=self.agent_id,
                    role="assistant",
                    content=policy_output,
                    is_state_end=False,
                )
            )

            if policy_output.strip() == "<no_deal/>":
                action = Proposal(no_deal=True, allocation_to_self=None)
            else:
                # parse per-type integers strictly
                alloc: Dict[str, int] = {}
                ok = True
                for t in observation.item_types:
                    m = re.search(fr"<{t}>([0-9]+)</{t}>", policy_output)
                    if not m:
                        ok = False
                        break
                    v = int(m.group(1))
                    if v < 0 or v > int(observation.stock.get(t, 0)):
                        ok = False
                        break
                    alloc[t] = v
                total_ok = sum(alloc.values()) == sum(int(observation.stock.get(t, 0)) for t in observation.item_types)
                if not ok or not total_ok:
                    action = Proposal(no_deal=True, allocation_to_self=None)
                else:
                    action = Proposal(no_deal=False, allocation_to_self=alloc)
        else:
            action = None

        agent_step_log = AgentActLog(
            chat_turns=self.state.chat_history[self.state.chat_counter :], info=None
        )
        self.state.chat_counter = len(self.state.chat_history)
        return action, agent_step_log

    def get_safe_copy(self):
        agent_copy = copy.copy(self)
        agent_copy.state = copy.deepcopy(self.state)
        return agent_copy

    def reset(self):
        self.state = DealNoDealAgentState(
            round_nb=0, nb_messages_sent_this_round=0, chat_counter=0, chat_history=[]
        )


