=================
Environments
=================

Multi-Agent Negotiation Environments require more features than gymnasium environments in order to be used as interfaces in general game running code. Differently than regular environments, LLM's output their action in a text format. Instead of having the parsing of the text done by the environment, we introduce another class called PlayerHandler. The player handler serves as an interface between the environment and the LLM. 

Taking inspiration from the `gymnasium <https://gymnasium.farama.org/>`_ library, we introduce a new standard for Multi-Agent Negotiation Environments. 

Our standard is based on the following features:

Environments are of the 
# TODO python abstract example 


```python
from abc import ABC, abstractmethod

class NegotiationEnvironment(ABC):

    def reset(self):
        """Reset the environment to an initial state and return the initial observation."""
        pass

    def step(self, actions):
        """Take a step in the environment using the provided actions.

        Args:
            actions (dict): A dictionary where keys are agent identifiers and values are actions.

        Returns:
            observation (dict): A dictionary where keys are agent identifiers and values are observations.
            reward (dict): A dictionary where keys are agent identifiers and values are rewards.
            done (bool): Whether the episode has ended.
            info (dict): Additional information about the environment.
        """
        pass

    def render(self):
        """Render the current state of the environment."""
        pass

    def close(self):
        """Perform any necessary cleanup."""
        pass
```
Similarly to the environment, the player handler is an abstract class that defines the following methods:

# TODO python abstract example 


```python
from abc import ABC, abstractmethod

class PlayerHandler(ABC):
    def get_action(self, observation):
        """Get an action from the player based on the observation."""
        pass

```








        
