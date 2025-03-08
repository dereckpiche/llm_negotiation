=================
Environments
=================

Multi-Agent Negotiation Environments require more features than gymnasium environments in order to be used as interfaces in general game running code. 
The two fundamental differences between gymnasium environments and Multi-Agent Negotiation Environments are:

1. Response from the LLM is a text action, not a discrete action. Therefore, appropriate parsing of the text is required. The model may need to be run multiple times to get the full action.
    This is why we introduce the `AgentHandler` class, which is responsible for parsing the LLM's response.
2. The environment needs to be able to handle multi-agent interactions.
    This is why we introduce the `NegotiationEnvironment` class, which is responsible for handling the multi-agent interactions.
3. MARL environments are complex to describe. In different contexts, the same environment may be described differently. Therefore, both the environement and the agent handlers are 
    responsible for describing a particular trajectory. This information is given by the `get_log_info` method. 
4. There might be a lot of overlap between the neural networks used by each agent. For instance, the same model may be used for all agents. This motivates a requirement for a
    policy identifier for each agent.

Taking inspiration from the `gymnasium <https://gymnasium.farama.org/>`_ library, we introduce a new standard for Multi-Agent Negotiation Environments. 

Our standard is based on the following features:

Environments are of the form:

.. code-block:: python

    class MarlEnvironment():

        def __init__(self):
            """Initialize the environment."""
            pass

        def reset(self):
            """Reset the environment to an initial state and return the initial observation.
            Returns:
                observation (dict): A dictionary where keys are agent identifiers and values are observations.
            """
            # (...)
            return observation

        def step(self, actions):
            """Take a step in the environment using the provided actions.

            Args:
                actions (dict): A dictionary where keys are agent identifiers and values are actions.

            Returns:
                observations (dict): A dictionary where keys are agent identifiers and values are observations.
                reward (dict): A dictionary where keys are agent identifiers and values are rewards.
                done (bool): Whether the episode has ended.
                info (dict): Additional information about the environment.
            """
            # (...)
            return observations, done, info

        def get_log_info(self):
            """Get additional information about the environment. This information is used to log the game.
            Returns:
                log_info (dict): Information about the environment required to log the game.
            """
            # (...)
            return log_info

        def render(self):
            """Render the current state of the environment."""
            pass

        def close(self):
            """Perform any necessary cleanup."""
            pass


    class AgentState():

        def __init__(self):
            """Initialize the agent state."""
            pass

        def step(self, observation_from_env, policy_output=None):
            """Update the agent state based on the observation and action. 
            The action is the output of the LLM.
            """

            Args:
                observation_from_env (dict): The observation of the environment. 
                policy_output : The output of the policy.

            Returns:
                policy_id (str): The policy identifier.
                policy_input (dict): The input to the policy.
                action : The official action to be sent to the environment.
                done (bool): Whether the LLM action is ready to be sent to the environment.
                info (dict): Additional information about the agent.
            """
            # (...)
            return policy_id, policy_input, action, done, info

        def get_log_info(self):
            """Get information about the agent required to log a trajectory.
            Returns:
                log_info (dict): Information about the agent required to log a trajectory.
            """ 
            # (...)
            return log_info

        def render(self):
            """Render the current state of the environment."""
            pass

        def close(self):
            """Perform any necessary cleanup."""
            pass


        

           


        

