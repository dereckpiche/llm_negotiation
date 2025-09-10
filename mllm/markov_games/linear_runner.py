import asyncio
import json
import os.path

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
    root = RolloutTreeRootNode(
        id=markov_game.get_id(),
        crn_id=markov_game.get_crn_id(),
        agent_ids=markov_game.get_agent_ids(),
    )
    previous_node = root
    while not terminated:
        terminated, step_log = await markov_game.step()
        current_node = RolloutTreeNode(step_log=step_log, time_step=time_step)
        previous_node.child = current_node
        previous_node = current_node
        time_step += 1

    return root
