import asyncio
import copy
from collections.abc import Callable
from dataclasses import dataclass

from mllm.markov_games.ipd.ipd_agent import IPDAgent
from mllm.markov_games.ipd.ipd_simulation import IPD
from mllm.markov_games.markov_game import MarkovGame
from mllm.markov_games.negotiation.dond_agent import DealNoDealAgent
from mllm.markov_games.negotiation.dond_simulation import DealNoDealSimulation
from mllm.markov_games.negotiation.no_press_nego_agent import NoPressAgent
from mllm.markov_games.negotiation.no_press_nego_simulation import NoPressSimulation
from mllm.markov_games.negotiation.tas_agent import TrustAndSplitAgent
from mllm.markov_games.negotiation.tas_rps_agent import TrustAndSplitRPSAgent
from mllm.markov_games.negotiation.tas_rps_simulation import TrustAndSplitRPSSimulation
from mllm.markov_games.negotiation.tas_simulation import TrustAndSplitSimulation
from mllm.markov_games.rollout_tree import (
    AgentActLog,
    RolloutTreeBranchNode,
    RolloutTreeNode,
    RolloutTreeRootNode,
    StepLog,
)
from mllm.markov_games.simulation import SimulationStepLog

AgentId = str


@dataclass
class AgentConfig:
    agent_id: str
    agent_name: str
    agent_class_name: str
    policy_id: str
    init_kwargs: dict


@dataclass
class MarkovGameConfig:
    id: int
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
    agents = {}
    for agent_config in config.agent_configs:
        agent_id = agent_config.agent_id
        agent_name = agent_config.agent_name
        agent_class = eval(agent_config.agent_class_name)
        agent = agent_class(
            seed=config.seed,
            agent_id=agent_id,
            agent_name=agent_name,
            policy=policies[agent_config.policy_id],
            **agent_config.init_kwargs,
        )
        agents[agent_id] = agent
    simulation = eval(config.simulation_class_name)(
        seed=config.seed,
        agent_ids=list(agents.keys()),
        **config.simulation_init_args,
    )
    markov_game = MarkovGame(
        id=config.id,
        crn_id=config.seed,
        agents=agents,
        simulation=simulation,
    )
    return markov_game
