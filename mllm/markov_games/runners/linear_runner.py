import asyncio
import json
import os.path
import uuid

from mllm.markov_games.markov_game import MarkovGame
from mllm.markov_games.rollout_tree import RolloutTreeNode, RolloutTreeRootNode


async def LinearRunner(
    markov_game: MarkovGame, output_folder: str
) -> RolloutTreeRootNode:
    """
    This method generates a trajectory without branching.
    """
    time_step = 0
    terminated = False
    root = RolloutTreeRootNode(id=int(str(uuid.uuid4().int)[:8]))
    previous_node = root
    while not terminated:
        terminated, step_log = await markov_game.step()
        current_node = RolloutTreeNode(step_log=step_log, time_step=time_step)
        previous_node.child = current_node
        previous_node = current_node
        time_step += 1

    # Export the tree & its schema
    os.makedirs(output_folder, exist_ok=True)
    export_path = os.path.join(
        output_folder, "mgid:" + str(root.id) + "_rollout_tree" + ".json"
    )
    with open(export_path, "w") as f:
        f.write(root.model_dump_json(indent=4))

    return root
    # TODO: export schema
