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
    "Welcome to an iterated game. You are {agent_name}.\n"
    "\n"
    "Setup:\n"
    "1. The game consists of multiple independent rounds.\n"
    "2. Each round, both agents are randomly assigned rock/paper/scissors hands. The assignment is always such that one hand is winning and the other hand is losing. There are no ties.\n"
    "3. You only see your own hand.\n"
    "4. The hands are not played against each other. They only determine the value of coins in that round:\n"
    "   - The agent with the winning hand has a coin value of 10.\n"
    "   - The agent with the losing hand has a coin value of 1.\n"
    "   - Winning hands are rock beats scissors, scissors beats paper, paper beats rock.\n"
    "   - Because assignments are random, over many rounds both agents are equally likely to have winning and losing hands.\n"
    "5. In each round, there are 10 coins to split between the two agents.\n"
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


class TrustAndSplitAgent(Agent):
    def __init__(
        self,
        seed: int,
        agent_id: str,
        policy: Callable[[List[Dict]], str],
        goal: str,
        num_message_chars: int,
    ):
        self.seed = seed
        self.agent_id = agent_id
        self.policy = policy
        self.goal = goal
        self.num_message_chars = num_message_chars
        self.state = TrustAndSplitAgentState(
            round_nb=0, nb_messages_sent_this_round=0, chat_counter=0, chat_history=[]
        )

    async def act(self, observation: TrustAndSplitObs) -> Tuple[Any, AgentActLog]:
        is_our_turn = observation.current_agent == self.agent_id
        action: Any = None
        round_nb = observation.round_nb

        prompt_parts: List[str] = []

        #######################################
        # build user prompt
        #######################################

        # First-ever call
        is_intro = round_nb == 0 and self.state.chat_counter == 0
        if is_intro:
            prompt_parts.append(
                INTRO_PROMPT.format(
                    agent_name=self.agent_id,
                    goal=self.goal,
                    quota_messages=observation.quota_messages_per_agent_per_round,
                )
            )

        # New round
        is_new_round = round_nb > self.state.round_nb
        if is_new_round or is_intro:
            self.state.nb_messages_sent_this_round = 0
            round_intro = ""
            if observation.last_hand_coagent is not None:
                round_intro += f"Last round, your hand was {observation.last_hand_agent}, and the other agent's hand was {observation.last_hand_coagent}. Based on these hands, your value per coin was {observation.last_value_agent}, while the other agent's value per coin was {observation.last_value_coagent}.\nYou proposed {observation.last_split_agent} coins and earned {round(observation.last_points_agent,1)} points, while the other agent proposed {observation.last_split_coagent} coins and earned {round(observation.last_points_coagent,1)} points.\n"
            round_intro += f"In this round, your hand is {observation.hand}.\n"
            prompt_parts.append(round_intro)
            self.state.round_nb = round_nb

        # Wait for message
        if not is_our_turn and not observation.split_phase:
            prompt_parts.append("Wait for other agent to send a message...")

        # Get last message
        if is_our_turn and not is_new_round and observation.last_message is not None:
            prompt_parts.append(f"Other agent said: {observation.last_message}")

        # Prompt to send message
        must_send_message = not observation.split_phase and is_our_turn
        if must_send_message:
            prompt_parts.append(
                f"Send your message now in <message>...</message> (<= {self.num_message_chars} chars)."
            )

        # Prompt to give split
        must_send_split = not must_send_message and observation.split_phase
        if must_send_split:
            prompt_parts.append(
                "Respond with <coins_to_self> x </coins_to_self> where x is an integer in [0, 10]."
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

        #######################################
        # Get policy action
        #######################################

        # Query policy for the appropriate format
        if must_send_message:
            return_regex = rf"<message>[\s\S]{{0,{self.num_message_chars}}}</message>"
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
        elif must_send_split:
            return_regex = r"<coins_to_self>\s*(10|[0-9])\s*</coins_to_self>"
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
                r"<coins_to_self>\s*(10|[0-9])\s*</coins_to_self>", policy_output
            )
            coins_int = int(m.group(1)) if m else int(policy_output)
            action = Split(coins_given_to_self=coins_int)
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
        self.state = TrustAndSplitAgentState(
            round_nb=0, nb_messages_sent_this_round=0, chat_counter=0, chat_history=[]
        )
