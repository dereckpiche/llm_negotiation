"""
In simple RL paradise, where the action dimensions are constant and well defined,
Agent classes are not necessary. But in MARL, with LLM's, there isn't always
a direct path from policy to action. For instance, from the observation of the environment,
a prompt must be created. Then, the outputs of the policy might be incorrect, so a second
request to the LLM must be sent before the action is well defined. This is why this Agent class exists.
It acts as a mini environment, bridging the gap between the core simulation and
the LLM policies.
"""

class Agent(object):

    def __init__(self):
        """
        Initialize the agent state.
        Usually, `policies` will be passed here.
        Policies is a dict that maps policy_id -> policy()
        """
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

    def seed(self):
        raise NotImplementedError

    def get_agent_info(self):
        raise NotImplementedError
