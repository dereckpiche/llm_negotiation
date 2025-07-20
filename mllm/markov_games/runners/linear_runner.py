import os.path
import asyncio
import json
from mllm.markov_games.markov_game import MarkovGame
from mllm.markov_games.rollout_tree import RolloutTreeNode, RolloutTreeRootNode


async def LinearRunner(
    markov_game: MarkovGame,
    output_folder: str
    ):
    """
    This method generates a trajectory without branching.
    """
    time_step = 0
    terminated = False
    root = RolloutTreeRootNode()
    previous_node = root
    while not terminated:
        terminated, step_log = await markov_game.step()
        current_node = RolloutTreeNode(step_log=step_log, time_step=time_step)
        previous_node.child = current_node
        previous_node = current_node

    # Export the tree & its schema
    os.makedirs(output_folder, exist_ok=True)
    export_path = os.path.join(output_folder, markov_game.id+".json")
    with open(export_path, "w") as f:
        f.write(root.model_dump_json(indent=4))

    # TODO: export schema
