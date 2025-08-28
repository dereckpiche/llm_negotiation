from collections.abc import Callable
from dataclasses import dataclass

from mllm.markov_games.ipd.ipd_agent import IPDAgent
from mllm.markov_games.ipd.ipd_simulation import IPD
from mllm.markov_games.markov_game import MarkovGame
from mllm.markov_games.negotiation.dond_agent import DealNoDealAgent
from mllm.markov_games.negotiation.dond_simulation import DealNoDealSimulation
from mllm.markov_games.negotiation.tas_agent import TrustAndSplitAgent
from mllm.markov_games.negotiation.tas_simulation import TrustAndSplitSimulation

AgentId = str


@dataclass
class AgentConfig:
    agent_class_name: str
    agent_id: AgentId
    policy_id: str
    init_kwargs: dict


@dataclass
class MarkovGameConfig:
    id: str
    seed: int
    simulation_class_name: str
    simulation_init_args: dict
    agent_configs: list[AgentConfig]


def init_markov_game_components(
    config: MarkovGameConfig, policies: dict[str, Callable[[list[dict]], str]]
):
    """
    TOWRITE
    """
    simulation_class = eval(config.simulation_class_name)
    simulation = simulation_class(seed=config.seed, **config.simulation_init_args)
    agents = {}
    for agent_config in config.agent_configs:
        agent_id = agent_config.agent_id
        agent_class = eval(agent_config.agent_class_name)
        agent = agent_class(
            seed=config.seed,
            agent_id=agent_id,
            policy=policies[agent_config.policy_id],
            **agent_config.init_kwargs,
        )
        agents[agent_id] = agent

    markov_game = MarkovGame(
        id=config.id,
        simulation=simulation,
        agents=agents,
    )
    return markov_game
