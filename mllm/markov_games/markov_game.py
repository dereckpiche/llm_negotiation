"""
This class unifies a simulation, and the agents acting in it (see `simulation.py` & `agent.py`).
In a MarkovGame step,
    1) each agent takes an action,
    2) the state transitions with respect to these actions,
    3) all relevant data of the step is appended to the historical data list

In order to perform 3), the agents and the simulation are expected, at each time step,
to return a log of the state transition (from their perspective).
For instance, the Simulation might send rewards and the agents might send prompting contexts to be used later to generate the training data.
A different approach would be to simply have the agents keep their data private and log it upon completion of a trajectory.
The approach we use here centralizes the data gathering aspect,
making it easy to create sub-trajectories (in the `runners` defined in `runners.py`) descriptions that
only log information for step transitions occuring after the branching out.
"""
from transformers.models.idefics2 import Idefics2Config
from mllm.markov_games.simulation import Simulation
from mllm.markov_games.agent import Agent
from copy import copy, deepcopy
import os, json

class MarkovGame(object):
    def __init__(
        self,
        id: str,
        simulation: type[Simulation],
        agents: dict[str, type[Agent]],
        output_path: str
    ):
        """
        Args:
            simulation:
                Simulation object. Example: IPDSimulation
            agents:
            output_path:
                Path where the step infos are saved.
        """
        self.id = id
        self.simulation = simulation
        self.agents = agents
        self.output_path = output_path
        self.agent_ids = self.agents.keys()
        self.simulation_step_infos = []
        self.agents_step_infos = {agent_id : [] for agent_id in self.agent_ids}
        self.actions = {}

    def set_action_of_agent(self, agent_id):
        """
        TOWRITE
        """
        agent = self.agents[agent_id]
        obs = self.simulation.get_obs_agent(agent_id)
        action, action_info = agent.act(observation=obs)
        self.actions[agent_id] = action
        self.agents_step_infos[agent_id].append(action_info)

    def unset_action_of_agent(self, agent_id):
        """
        TOWRITE
        """
        self.actions[agent_id] = None
        self.agents_step_infos[agent_id].pop()

    def set_actions(self):
        """
        TOWRITE
        """
        for agent_id in self.agent_ids:
            self.set_action_of_agent(agent_id)

    def unset_actions(self):
        """
        TOWRITE
        """
        for agent_id in self.agent_ids:
            self.unset_action_of_agent(agent_id)

    def take_simulation_step(self):
        """
        TOWRITE
        """
        terminated, simulation_step_info = self.simulation.step(self.actions)
        self.simulation_step_infos.append(simulation_step_info)
        return terminated

    async def step(self):
        """
        TOWRITE
        """
        self.set_actions()
        terminated = self.take_simulation_step()
        return terminated

    def get_new_branch(self):
        """
        TOWRITE
        """
        # Only deep copy the states. We don't want to deep copy policies, for instance.
        # TODO: add different id!
        new_markov_game = copy(self)
        new_markov_game.simulation_step_infos = []
        new_markov_game.agents_step_infos = {agent_id : [] for agent_id in self.agent_ids}
        new_markov_game.simulation.state = deepcopy(new_markov_game.simulation.state)
        for agent in new_markov_game.agents:
            agent.state = deepcopy(agent.state)
        return new_markov_game

    def run(self):
        """
        Runs the markov game
        """
        terminated = False
        while not terminated:
            terminated = self.step()

    def export(self):
        """
        Exports the step infos. At the specified path.
        """
        simulation_out_path = os.path.join(self.output_path, "simulation")
        with open(simulation_out_path, "w") as f:
            json.dump(self.simulation_step_infos, f)
        for agent_id in self.agent_ids:
            agent_out_path = os.path.join(self.output_path, agent_id)
            with open(agent_out_path, "w") as f:
                agent_step_infos =  self.agents_step_infos[agent_id]
                json.dump(agent_step_infos, f)
