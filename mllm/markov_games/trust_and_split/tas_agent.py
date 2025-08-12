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


@dataclass
class TrustAndSplitAgentState:
    round_nb: int
    nb_messages_sent_this_round: int
    chat_counter: int
    chat_history: List[ChatTurn]


INTRO_PROMPT = (
    "Welcome to an iterated game. You are {agent_name}. "
    "Each round there are 10 coins. Your per-coin value for the current round will be provided. "
    "Agents can exchange short messages and then each proposes how many coins they keep for themselves. "
    "If totals exceed 10, coins are allocated proportionally. "
    "Message format: <message>...</message> (<=400 chars). "
    "Split format: <coins_to_self>x</coins_to_self>."
    "Your goal is: {goal}"
)


class TrustAndSplitAgent(Agent):
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
        self.nb_messages_per_round = int(nb_messages_per_round)
        self.goal = goal
        self.state = TrustAndSplitAgentState(
            round_nb=0, nb_messages_sent_this_round=0, chat_counter=0, chat_history=[]
        )

    async def act(self, observation: TrustAndSplitObs) -> Tuple[Any, AgentActLog]:
        action: Any = None
        round_nb = observation.round_nb

        # Build a single user prompt per call to satisfy logging constraint
        prompt_parts: List[str] = []

        # First-ever call
        if round_nb == 0 and self.state.chat_counter == 0:
            prompt_parts.append(
                INTRO_PROMPT.format(agent_name=self.agent_id, goal=self.goal)
            )

        # New round
        if round_nb > self.state.round_nb:
            self.state.nb_messages_sent_this_round = 0
            round_intro = (
                f"New round {round_nb}. Your value per coin: {observation.value}. "
                f"Last round, other agent value per coin: {observation.last_value_coagent}."
            )
            prompt_parts.append(round_intro)
            self.state.round_nb = round_nb

        # Turn-dependent instruction
        is_our_turn = observation.current_agent == self.agent_id
        if is_our_turn:
            if (
                not observation.first_split_done
                and self.state.nb_messages_sent_this_round < self.nb_messages_per_round
            ):
                instr = (
                    f"Other agent said: {observation.last_message}\n"
                    f"Send your message now in <message>...</message> (<=400 chars)."
                )
                prompt_parts.append(instr)
            else:
                instr = (
                    f"Other agent said: {observation.last_message}\n"
                    f"Finalize: respond as <coins_to_self>x</coins_to_self> where x is an integer in [0, 10]."
                )
                prompt_parts.append(instr)
        else:
            prompt_parts.append(
                f"Waiting; it's {observation.current_agent}'s turn. Last message: {observation.last_message}"
            )

        # Append one ChatTurn with is_state_end=True
        user_prompt = "\n".join(prompt_parts)
        self.state.chat_history.append(
            ChatTurn(
                agent_id=self.agent_id,
                role="user",
                content=user_prompt,
                is_state_end=True,
            )
        )

        # If it's our turn, query policy for the appropriate format
        if is_our_turn:
            if (
                not observation.first_split_done
                and self.state.nb_messages_sent_this_round < self.nb_messages_per_round
            ):
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
            else:
                return_regex = r"<coins_to_self>([0-9]+)</coins_to_self>"
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
                import re as _re

                m = _re.search(
                    r"<coins_to_self>([0-9]+)</coins_to_self>", policy_output
                )
                coins_int = int(m.group(1)) if m else int(policy_output)
                action = Split(coins_given_to_self=coins_int)

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
        self.state = TrustAndSplitAgentState(
            round_nb=0, nb_messages_sent_this_round=0, chat_counter=0, chat_history=[]
        )
