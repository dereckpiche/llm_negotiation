from typing import Any, Dict, List, Tuple

from mllm.markov_games.rollout_tree import AgentActLog, ChatTurn
from mllm.markov_games.negotiation.nego_agent import NegotiationAgent, NegotiationAgentState
from mllm.markov_games.negotiation.no_press_nego_simulation import DeterministicNoPressObs
from mllm.markov_games.negotiation.nego_simulation import Split


class DeterministicNoPressAgent(NegotiationAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # No communication in this variant
        self.quota_messages_per_agent_per_round = 0
        self.intro_prompt = (
            "Deterministic no-press split. You are {agent_id}. "
            "You see both values and there is no messaging. "
            "Respond only with <coins_to_self>x</coins_to_self>."
        )
        self.new_round_prompt = (
            "Round {round_nb}. My value: {value}, Other value: {other_value}. Max coins: {quantities[coins]}."
        )
        self.send_split_prompt = (
            "Submit your split as <coins_to_self>([0-9]+)</coins_to_self>."
        )

    def get_message_regex(self, observation: DeterministicNoPressObs) -> str:
        return r"^$"  # No messages allowed

    def get_split_regex(self, observation: DeterministicNoPressObs) -> str:
        max_coins = observation.quantities["coins"]
        return rf"<coins_to_self>({('|'.join(str(i) for i in range(0, max_coins+1)))})</coins_to_self>"

    def get_split_action(self, policy_output: str, observation: DeterministicNoPressObs) -> Split:
        import re as _re
        m = _re.search(r"<coins_to_self>([0-9]+)</coins_to_self>", policy_output)
        coins_int = int(m.group(1)) if m else int(policy_output)
        return Split(items_given_to_self={"coins": coins_int})


