import copy
from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from mllm.markov_games.agent import Agent
from mllm.markov_games.rollout_tree import AgentActLog, ChatTurn
from mllm.markov_games.negotiation.nego_simulation import (
    Message,
    Split,
    NegotiationObs,
)

@dataclass
class NegotiationAgentState:
    round_nb: int
    nb_messages_sent_this_round: int
    chat_counter: int
    chat_history: List[ChatTurn]


class NegotiationAgent(Agent):
    def __init__(
        self,
        seed: int,
        agent_id: str,
        policy: Callable[[List[Dict]], str],
        goal: str,
        item_types: List[str],
        quota_messages_per_agent_per_round: int,
    ):
        self.seed = seed
        self.agent_id = agent_id
        self.policy = policy
        self.goal = goal
        self.item_types = item_types
        self.quota_messages_per_agent_per_round = quota_messages_per_agent_per_round
        self.state = NegotiationAgentState(
            round_nb=0, nb_messages_sent_this_round=0, chat_counter=0, chat_history=[]
        )

        # Implemented in variants
        self.intro_prompt = (
            "Welcome to an iterated game. You are {current_agent}.\n"
            "(...) describe rules here ..."
            "Your goal: {goal}"
        ).format(current_agent=self.agent_id, goal=self.goal, quota_messages=self.quota_messages_per_agent_per_round, item_types=self.item_types)
        self.new_round_prompt = ""
        self.last_round_prompt = ""
        self.send_split_prompt = ""


        self.wait_for_message_prompt = (
            "Wait for {other_agent_id} to send a message..."
        )
        self.last_message_prompt = (    
            "{other_agent_id} said: {last_message}"
        )
        self.send_message_prompt = (
            "Send your message now in <message>...</message> (<=400 chars). "
        )

    @abstractmethod
    def get_message_regex(self, observation: NegotiationObs) -> str:
        pass

    @abstractmethod
    def get_split_regex(self, observation: NegotiationObs) -> str:
        pass

    @abstractmethod
    def get_split_action(self, policy_output: str, observation: NegotiationObs) -> Split:
        pass

    async def act(self, observation: NegotiationObs) -> Tuple[Any, AgentActLog]:
        is_our_turn = observation.current_agent == self.agent_id
        action: Any = None
        round_nb = observation.round_nb

        prompt_parts: List[str] = []
        obs_ctx = vars(observation)

        #######################################
        # build user prompt
        #######################################

        # First-ever call
        is_intro = round_nb == 0 and self.state.chat_counter == 0
        if is_intro:
            prompt_parts.append(self.intro_prompt.format(goal=self.goal, quota_messages=observation.quota_messages_per_agent_per_round, item_types=self.item_types, **obs_ctx))

        # New round
        is_new_round = round_nb > self.state.round_nb
        if is_new_round:
            self.state.nb_messages_sent_this_round = 0
            if not is_intro:
                prompt_parts.append(self.last_round_prompt.format(**obs_ctx))
            prompt_parts.append(self.new_round_prompt.format(round_nb=round_nb, **obs_ctx))
            self.state.round_nb = round_nb

        # Wait for message
        if not is_our_turn and not observation.split_phase:
            prompt_parts.append(self.wait_for_message_prompt.format(round_nb=round_nb, **obs_ctx))

        # Get last message
        if is_our_turn and not is_new_round:
            prompt_parts.append(self.last_message_prompt.format(round_nb=round_nb, **obs_ctx))

        # Prompt to send message
        must_send_message = (
            not observation.split_phase
            and is_our_turn
        )
        if must_send_message:
            prompt_parts.append(
                self.send_message_prompt.format(round_nb=round_nb, **obs_ctx)
            )

        # Prompt to give split
        must_send_split = not must_send_message and observation.split_phase
        if must_send_split:
            prompt_parts.append(
                self.send_split_prompt.format(round_nb=round_nb, **obs_ctx)
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
            return_regex = self.get_message_regex(observation)
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
            return_regex = self.get_split_regex(observation)
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
            action = self.get_split_action(policy_output, observation)
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
        self.state = NegotiationAgentState(
            round_nb=0, nb_messages_sent_this_round=0, chat_counter=0, chat_history=[]
        )
