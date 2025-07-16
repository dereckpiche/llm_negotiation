
from types import ClassMethodDescriptorType
from mllm.markov_games.simulation import Simulation
from mllm.markov_games.agent import Agent
from mllm.markov_games.markov_game import MarkovGame
import os
import json
from copy import copy, deepcopy
from dataclasses import dataclass
from collections.abc import Callable
from mllm.markov_games.ipd.ipd_agent import IPDAgent
from mllm.markov_games.ipd.ipd_simulation import IPD

@dataclass
class AgentConfig:
    agent_class_name: str
    agent_id: int
    policy_id: str
    init_kwargs: dict

@dataclass
class MarkovGameConfig:
    id: str
    seed: int
    simulation_class_name: str
    simulation_init_args: dict
    agent_configs: list[AgentConfig]
    output_path: str

def init_markov_game_components(
    config: MarkovGameConfig,
    policies: dict[str, Callable[[list[dict]], str]]
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
            seed = config.seed,
            agent_id = agent_id,
            policy = policies[agent_config.policy_id],
            **agent_config.init_kwargs
        )
        agents[agent_id] = agent

    markov_game = MarkovGame(
        id = id,
        simulation=simulation,
        agents=agents,
        output_path=config.output_path
    )
    return markov_game
