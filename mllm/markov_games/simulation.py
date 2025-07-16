"""
A Simulation is the environment of a Markov Game.
The Simulation is not responsible for properly checking / formatting the responses of LLM's.
This is the job of the `Agent` class.
Simulations expect clean actions, and are defined similarly to `gymnasium` environments, except that they are adapted for the Multi-agent setting.
"""

from abc import ABC, abstractmethod
from numpy.random import default_rng

class Simulation(ABC):

    @abstractmethod
    def __init__(self, seed: int, *args, **kwargs):
        self.seed = seed
        self.rng = default_rng(self.seed)

    def step(self, actions):
        """ Returns terminated, info

        Returns:
            Infos:

        """
        raise NotImplementedError

    def get_obs(self):
        """ Returns all agent observations in dict

        Returns:
            observations
        """
        raise NotImplementedError

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        raise NotImplementedError

    def get_obs_size(self):
        """ Returns the shape of the observation """
        raise NotImplementedError

    def get_state(self):
        raise NotImplementedError

    def get_state_size(self):
        """ Returns the shape of the state"""
        raise NotImplementedError

    def get_avail_actions(self):
        raise NotImplementedError

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        raise NotImplementedError

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        raise NotImplementedError

    def reset(self):
        """ Returns initial observations and states"""
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def seed(self):
        raise NotImplementedError

    def save_replay(self):
        raise NotImplementedError

    def get_simulation_info(self):
        raise NotImplementedError
