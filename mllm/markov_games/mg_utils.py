from collections.abc import Callable
from dataclasses import dataclass
import copy
import asyncio

from mllm.markov_games.markov_game import MarkovGame
from mllm.markov_games.ipd.ipd_agent import IPDAgent
from mllm.markov_games.ipd.ipd_simulation import IPD
from mllm.markov_games.trust_and_split.tas_agent import TrustAndSplitAgent
from mllm.markov_games.trust_and_split.tas_simulation import TrustAndSplitSimulation
from mllm.markov_games.deal_no_deal.dond_agent import DealNoDealAgent
from mllm.markov_games.deal_no_deal.dond_simulation import DealNoDealSimulation

from mllm.markov_games.markov_game import MarkovGame
from mllm.markov_games.rollout_tree import RolloutTreeRootNode, StepLog, RolloutTreeBranchNode
from mllm.markov_games.rollout_tree import AgentActLog
from mllm.markov_games.simulation import SimulationStepLog
from mllm.markov_games.rollout_tree import RolloutTreeNode

AgentId = str

def stop_when_round_ends(step_log: StepLog) -> bool:
    """
    Simplest stop condition. Will return True if step log is the last time step of a round.
    This will throw an error if this information is not available in the simulation info.
    """
    # TODO
    

def group_time_steps(rollout_tree: RolloutTreeRootNode, accumulation_stop_condition: Callable[[StepLog], bool]) -> RolloutTreeRootNode:
    """
    During generation, we create rollout trees according to the real time steps.
    However, during training, we might want to treat groups of time steps as a single time step.
    As a concrete example, take Trust-and-Split. At each round, say we have X time steps of communication and then one time step for the split. 
    Then the communication actions will not get any reward, and the split action will get the reward. During REINFORCE training, with discounting, this 
    can cause training instability. We could instead treat every action in the round as being part of a single action, and give it the reward of the split action.
    This method helps to do this sort of grouping. 
    It accumulates actions until the accumulation_stop_condition is met, and then creates a new node with the accumulated actions.
    It then recursively calls itself on the child node.
    Details:
    - The reward for the group is the reward of the last time step in the group.
    - The simulation log for the group is the simulation log of the last time step in the group.
    - The state end for the group becomes the first state end in the group.
    - The agent info for the group is the agent info of the last time step in the group.
    """
    def group_step_logs(step_logs: list[StepLog]) -> StepLog:
        """
        Concatenate per-agent chat turns across steps; keep only the first is_state_end.
        """
        last_sim_log = step_logs[-1].simulation_step_log
        agent_ids = {aid for s in step_logs for aid in s.action_logs.keys()}
        grouped_logs: dict[AgentId, AgentActLog] = {}
        for aid in agent_ids:
            turns = []
            for s in step_logs:
                act = s.action_logs.get(aid)
                if act and act.chat_turns:
                    turns.extend(copy.deepcopy(act.chat_turns))
            put_state_end_false = False
            # Only the first state_end should be True, the rest should be False
            for t in turns:
                if t.is_state_end: 
                    put_state_end_false = True
                    continue
                if put_state_end_false: t.is_state_end = False
            grouped_logs[aid] = AgentActLog(chat_turns=turns, info=step_logs[-1].action_logs[aid].info) 
        return StepLog(action_logs=grouped_logs, simulation_step_log=last_sim_log)

    def group_time_steps_rec(current_node: RolloutTreeNode | RolloutTreeBranchNode , group_time_step: int, accumulation_step_logs: list[StepLog]) -> RolloutTreeNode | RolloutTreeBranchNode:
        """
        Groups time steps. Recursion is used to handle branches.
        """
        current_group_node = first_group_node
        first_group_node = None
        while not isinstance(current_node, RolloutTreeBranchNode) and current_node is not None:

            # Special recursive case for branches
            if isinstance(current_node, RolloutTreeBranchNode):
                main_child_group_node = group_time_steps_rec(
                    current_node=current_node.main_child, 
                    group_time_step=group_time_step, 
                    accumulation_step_logs=copy.deepcopy(accumulation_step_logs))
                branches = {}
                for agent_id, branch_nodes in current_node.branches.items():
                    branch_group_nodes = []
                    for branch_node in branch_nodes:
                        branch_group_node = group_time_steps_rec(
                            current_node=branch_node, 
                            group_time_step=group_time_step, 
                            accumulation_step_logs=copy.deepcopy(accumulation_step_logs))
                        branch_group_nodes.append(branch_group_node)
                    branches[agent_id] = branch_group_nodes
                return RolloutTreeBranchNode(main_child=main_child_group_node, branches=branches)

            # Accumulate
            accumulation_step_logs.append(current_node.step_log)
            if accumulation_stop_condition(current_node.step_log, current_node.time_step):
                grouped_step_logs = group_step_logs(accumulation_step_logs)
                accumulation_step_logs = []
                new_group_node = RolloutTreeNode(step_log=grouped_step_logs, time_step=group_time_step, child=None)
                if first_group_node == None:
                    first_group_node = new_group_node
                time_step += 1
            current_group_node.child = new_group_node
            current_group_node = new_group_node
            current_node = current_node.child
        return first_group_node

    node = group_time_steps_rec(current_node=rollout_tree.child, group_time_step=0, accumulation_step_logs=[])
    return RolloutTreeRootNode(id=rollout_tree.id, child=node)


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




