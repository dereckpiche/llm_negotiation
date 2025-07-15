
from types import ClassMethodDescriptorType
from mllm.markov_games.simulation import Simulation
from mllm.markov_games.agent import Agent
from mllm.markov_games.markov_game import MarkovGame
import os
import json
from copy import copy, deepcopy
from dataclasses import dataclass
from collections.abc import Callable


@dataclass
class AgentConfig:
    agent_id: int
    agent_class: str
    policy_id: str
    init_kwargs: dict

@dataclass
class MarkovGameConfig:
    simulation_class: str
    simulation_init_args: dict
    agent_configs: list[AgentConfig]
    output_path: str

def hydra_conf_to_mg_config(conf:dict):
    simulation_class: ClassType
    simulation_init_args: dict
    agent_configs: list[AgentConfig]

def init_markov_game_components(
    config: MarkovGameConfig,
    policies: dict[str, Callable[[list[dict]], str]]
    ):
    """
    TOWRITE
    """
    simulation = eval(config.simulation_class)(**config.simulation_init_args)
    agents = {}
    for agent_config in config.agent_configs:
        agent_id = agent_config.agent_id
        agent_class = eval(agent_config.agent_class)
        agent = agent_class(
            agent_id = agent_id,
            policy = policies[agent_config.policy_id],
            **agent_config.init_kwargs
        )
        agents[agent_id] = agent
    markov_game = MarkovGame(
        simulation=simulation,
        agents=agents,
    )
    return markov_game
