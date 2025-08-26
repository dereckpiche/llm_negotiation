import asyncio
import copy
import json
import os.path
from typing import Any, Tuple

from mllm.markov_games.markov_game import AgentAndActionSafeCopy, MarkovGame
from mllm.markov_games.rollout_tree import (
    AgentActLog,
    RolloutTreeBranchNode,
    RolloutTreeNode,
    RolloutTreeRootNode,
    StepLog,
)

AgentId = str
import uuid


async def run_with_unilateral_alt_action(
    markov_game: MarkovGame,
    agent_id: AgentId,
    time_step: int,
    branch_node: RolloutTreeBranchNode,
    max_depth: int,
):
    """
    This function is used to generate a new branch for a given agent.
    """

    # Generate alternative action and take a step
    await markov_game.set_action_of_agent(agent_id)
    terminated: bool = markov_game.take_simulation_step()
    step_log = markov_game.get_step_log()
    first_alternative_node = RolloutTreeNode(
        step_log=step_log,
        time_step=time_step,
    )

    # Generate rest of trajectory up to max depth
    time_step += 1
    counter = 1
    previous_node = first_alternative_node
    while not terminated and counter <= max_depth:
        terminated, step_log = await markov_game.step()
        current_node = RolloutTreeNode(step_log=step_log, time_step=time_step)
        previous_node.child = current_node
        previous_node = current_node
        counter += 1
        time_step += 1

    if branch_node.branches == None:
        branch_node.branches = {agent_id: [first_alternative_node]}
    else:
        agent_branches = branch_node.branches.get(agent_id, [])
        agent_branches.append(first_alternative_node)
        branch_node.branches[agent_id] = agent_branches


async def AlternativeActionsRunner(
    markov_game: MarkovGame,
    output_folder: str,
    nb_alternative_actions: int,
    max_depth: int,
):
    """
    This method generates a trajectory with partially completed branches,
    where the branching comes from taking unilateraly different actions.
    The resulting data is used to estimate the updated advantage alignment policy gradient terms.
    Let k := nb_sub_steps. Then the number of steps generated is O(Tk), where T is
    the maximum trajectory length.
    """

    tasks = []
    time_step = 0
    terminated = False
    root = RolloutTreeRootNode(
        id=int(str(uuid.uuid4().int)[:8]), rng_seed=markov_game.get_rng_seed()
    )
    previous_node = root

    while not terminated:
        mg_before_action = markov_game.get_safe_copy()

        # Get safe copies for main branch
        agent_action_safe_copies: dict[
            AgentId, AgentAndActionSafeCopy
        ] = await markov_game.get_actions_of_agents_without_side_effects()

        markov_game.set_actions_of_agents_manually(agent_action_safe_copies)
        terminated = markov_game.take_simulation_step()
        main_node = RolloutTreeNode(
            step_log=markov_game.get_step_log(), time_step=time_step
        )
        branch_node = RolloutTreeBranchNode(main_child=main_node)
        previous_node.child = branch_node
        previous_node = main_node

        # Get alternative branches by generating new unilateral actions
        for agent_id in markov_game.agent_ids:
            for _ in range(nb_alternative_actions):
                # Get safe copies for branches
                branch_agent_action_safe_copies: dict[
                    AgentId, AgentAndActionSafeCopy
                ] = {
                    agent_id: AgentAndActionSafeCopy(
                        action=copy.deepcopy(agent_action_safe_copy.action),
                        action_info=copy.deepcopy(agent_action_safe_copy.action_info),
                        agent_after_action=agent_action_safe_copy.agent_after_action.get_safe_copy(),
                    )
                    for agent_id, agent_action_safe_copy in agent_action_safe_copies.items()
                }
                mg_branch: MarkovGame = mg_before_action.get_safe_copy()
                other_agent_id = [id for id in mg_branch.agent_ids if id != agent_id][0]
                mg_branch.set_action_and_agent_after_action_manually(
                    agent_id=other_agent_id,
                    agent_action_safe_copy=branch_agent_action_safe_copies[
                        other_agent_id
                    ],
                )
                task = asyncio.create_task(
                    run_with_unilateral_alt_action(
                        markov_game=mg_branch,
                        time_step=time_step,
                        agent_id=agent_id,
                        branch_node=branch_node,
                        max_depth=max_depth,
                    )
                )
                tasks.append(task)
        time_step += 1

    # wait for all branches to complete
    await asyncio.gather(*tasks)

    # Export the tree & its schema
    os.makedirs(output_folder, exist_ok=True)
    export_path = os.path.join(
        output_folder, "mgid:" + str(root.id) + "_rollout_tree" + ".json"
    )
    with open(export_path, "w") as f:
        f.write(root.model_dump_json(indent=4))

    return root
