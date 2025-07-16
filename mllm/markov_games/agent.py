"""
In simple RL paradise, where the action dimensions are constant and well defined,
Agent classes are not necessary. But in MARL, with LLM's, there isn't always
a direct path from policy to action. For instance, from the observation of the environment,
a prompt must be created. Then, the outputs of the policy might be incorrect, so a second
request to the LLM must be sent before the action is well defined. This is why this Agent class exists.
It acts as a mini environment, bridging the gap between the core simulation and
the LLM policies.
"""

from abc import ABC, abstractmethod
from numpy.random import default_rng
from collections.abc import Callable

class Agent(ABC):

    @abstractmethod
    def __init__(self, seed: int, agent_id:int, policy: Callable[[list[dict]], str], *args, **kwargs):
        """
        Initialize the agent state.
        """
        self.seed = seed
        self.agent_id = agent_id
        self.policy = policy
        self.rng = default_rng(self.seed)
        raise NotImplementedError

    async def act(self, observation):
        """
        Query (possibly multiple times) a policy (or possibly a pool of policies) to
        obtain the action of the agent.

        Example:
        action = None
        prompt = self.observation_to_prompt(observation)
        while not self.valid(action):
            output = await self.policy.generate(prompt)
            action = self.policy_output_to_action(output)
        return action

        Returns:
            action:
                The action of the agent. Example: "defect"
            info:
                Data of agent perspective at time step.
                Used to generate training data later.
                Example: [{"role":user, "content":"Other player played cooperate.",
                {"role":"assistant", "content":"defect"}]
        """
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError



    def get_agent_info(self):
        raise NotImplementedError
