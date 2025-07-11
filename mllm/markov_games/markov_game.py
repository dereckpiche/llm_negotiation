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

from types import ClassMethodDescriptorType


class MarkovGame(object):
    def init(
        self,
        simulation: Simulation,
        agents: dict[str, Agent],
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
        self.simulation = simulation
        self.agents = agents
        self.agent_ids = self.agents.keys()
        self.simulation_step_infos = []
        self.agents_step_infos = {agent_id : [] for agent_id in self.agent_ids}
        self.actions = {}

    def set_action_of_agent(self, agent_id):
        agent = self.agents[agent_id]
        obs = self.simulation.get_obs_agent(agent_id)
        action, action_info = agent.act(obs)
        self.actions[agent_id] = action
        self.agents_step_infos[agent_id].append(action_info)

    def unset_action_of_agent(self, agent_id):
        self.actions[agent_id] = None
        self.agents_step_infos[agent_id].pop()

    def set_actions(self):
        for agent_id in self.agent_ids:
            self.set_action_of_agent(agent_id)

    def unset_actions(self):
        for agent_id in self.agent_ids:
            self.unset_action_of_agent(agent_id)

    def take_simulation_step(self):
        terminated, simulation_step_info = self.simulation.step(self.actions)
        self.simulation_step_infos.append(simulation_step_info)
        return terminated

    async def step(self):
        # get action of each agents
        self.set_actions()
        terminated = self.take_simulation_step()
        # TODO: add boolean check that new actions were added
        #

    def get_new_branch(self)
        # TODO copy everything except policies and step infos
        # keep only step infos of last time step
        # return markov_game_copy
        raise NotImplementedError

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
        raise NotImplementedError
