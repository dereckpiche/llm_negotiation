import os.path
import asyncio
import json
from mllm.markov_games.markov_game import MarkovGame
from mllm.markov_games.rollout_tree import RolloutTreeNode, RolloutTreeRootNode, RolloutTreeBranchNode, StepLog
AgentId = str
import uuid

async def run_with_unilateral_alt_action(
        markov_game: MarkovGame,
        agent_id: AgentId,
        branch_id: int,
        time_step: int,
        branch_node: RolloutTreeBranchNode):
        """
        This function is used to generate a new branch for a given agent.
        """

        # Generate new action and take a step
        markov_game.unset_action_of_agent(agent_id)
        await markov_game.set_action_of_agent(agent_id)
        terminated = markov_game.take_simulation_step()
        step_log = markov_game.get_step_log()
        first_alternative_node = RolloutTreeNode(
            step_log=step_log,
            time_step=time_step,
        )

        # Generate rest of trajectory up to max depth
        time_step += 1
        counter = 1
        previous_node = first_alternative_node
        while not terminated and counter < depth:
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
    nb_alternative_actions: int = 1,
    depth: int = 1):
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
    root = RolloutTreeRootNode(id = int(str(uuid.uuid4().int)[:8]))
    previous_node = root

    while not terminated:
        terminated, step_log = await markov_game.step()
        main_node = RolloutTreeNode(step_log=step_log, time_step=time_step)
        branch_node = RolloutTreeBranchNode(main_child=main_node)
        previous_node.child = branch_node
        previous_node = main_node

        # Get alternative branches by generating new unilateral actions
        for agent_id in markov_game.agent_ids:
            for _ in range(nb_alternative_actions):
                mg_branch = markov_game.get_safe_copy()
                # branch_name = f"alternate_action_of_{agent_id}_time_{time_step}"
                # branch_info = BranchNodeInfo(
                #     branch_id = branch_name,
                #     branch_for = agent_id,
                #     branch_type = "unilateral_deviation"
                # )
                branch_id = int(str(uuid.uuid4().int)[:8])
                task = asyncio.create_task( run_with_unilateral_alt_action(
                    markov_game = mg_branch,
                    time_step = time_step,
                    agent_id = agent_id,
                    branch_id = branch_id,
                    branch_node = branch_node,
                ) )
                tasks.append(task)
        time_step += 1

    # wait for all branches to complete
    await asyncio.gather(*tasks)

    # Export the tree & its schema
    os.makedirs(output_folder, exist_ok=True)
    export_path = os.path.join(output_folder, markov_game.id+".json")
    with open(export_path, "w") as f:
        f.write(root.model_dump_json(indent=4))

    return root
