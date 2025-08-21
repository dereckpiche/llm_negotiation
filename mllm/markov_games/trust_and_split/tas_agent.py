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


class TrustAndSplitAgent(Agent):
    def __init__(
        self,
        seed: int,
        agent_id: str,
        policy: Callable[[List[Dict]], str],
        goal: str,
    ):
        self.seed = seed
        self.agent_id = agent_id
        self.policy = policy
        self.goal = goal
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
                INTRO_PROMPT.format(agent_name=self.agent_id, goal=self.goal, quota_messages=observation.quota_messages_per_agent_per_round)
            )

        # New round
        is_new_round = round_nb > self.state.round_nb
        if is_new_round or is_intro:
            self.state.nb_messages_sent_this_round = 0
            round_intro = (
                f"New round {round_nb}. Your hand: {observation.hand}. "
            )
            if observation.last_hand_coagent is not None:
                round_intro += f"Last round, other agent's hand: {observation.last_hand_coagent}. "
            prompt_parts.append(round_intro)
            self.state.round_nb = round_nb

        # Wait for message
        if not is_our_turn and not observation.split_phase:
            prompt_parts.append("Wait for other agent to send a message...")

        # Get last message
        if is_our_turn and not is_new_round:
            prompt_parts.append(f"Other agent said: {observation.last_message}")

        # Prompt to send message
        must_send_message = (
            not observation.split_phase
            and is_our_turn
        )
        if must_send_message:
            prompt_parts.append(
                "Send your message now in <message>...</message> (<=400 chars). "
            )

        # Prompt to give split
        must_send_split = not must_send_message and observation.split_phase
        if must_send_split:
            prompt_parts.append(
                "Respond with <coins_to_self>x</coins_to_self> where x is an integer in [0, 10]."
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
        elif must_send_split:
            return_regex = r"<coins_to_self>(10|[0-9])</coins_to_self>"
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

            m = _re.search(r"<coins_to_self>([0-9]+)</coins_to_self>", policy_output)
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
