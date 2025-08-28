from mllm.markov_games.negotiation.nego_agent import NegotiationAgent
from mllm.markov_games.negotiation.nego_simulation import Split


class TrustAndSplitAgent(NegotiationAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO: take inspiration from DOND prompt
        self.intro_prompt = (
            "TrustAndSplit negotiation over coins. You are {agent_name}. Your per-coin value is revealed, "
            "the other agent may disclose theirs via chat. 10 coins to split."
        )
        self.new_round_prompt = "Round {round_nb}. Your per-coin value: {value}. There are {quantities[coins]} coins."
        self.send_split_prompt = "Respond with <coins_to_self>x</coins_to_self> where x is an integer in [0, {quantities[coins]}]."

    def get_message_regex(self, observation) -> str:
        return r"<message>[\s\S]{0,400}</message>"

    def get_split_regex(self, observation) -> str:
        max_coins = int(observation.quantities["coins"])
        return rf"<coins_to_self>({('|'.join(str(i) for i in range(0, max_coins+1)))})</coins_to_self>"

    def get_split_action(self, policy_output: str, observation) -> Split:
        import re as _re

        m = _re.search(r"<coins_to_self>([0-9]+)</coins_to_self>", policy_output)
        coins_int = int(m.group(1)) if m else int(policy_output)
        return Split(items_given_to_self={"coins": coins_int})
