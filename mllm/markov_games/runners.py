"""
These methods allow non-linear trajectory generation. Trajectories may
branch out into sub-trajectories at different states according to different schemes.
This type of generation is required for AdAlign* and VinePPO, for instance.

At a high level, these methods should export a bunch of trajectory json files and a
tree structure. The trajectory files should not overlap/repeat themselves. For instance,
if a sub-trajectory `st` branches out from `t` at time step `i`, then the file of
`st` should only contain historical information from `i` to `T` (terminal time step).
The implicit standardization of the book keeping of step information in `markov_game.py`
helps following this constraint.
"""

from anytree import Node, RenderTree
from anytree.exporter import DotExporter
from os import path
from _typeshed import NoneType
import asyncio

async def AlternativeActionsRunner(
    markov_game: MarkovGame,
    nb_alternative_actions: int = 2,
    nb_sub_steps: int = 1)
    """
    This method generates a trajectory with partially completed branches,
    where the branching comes from taking unilateraly different actions.
    The resulting data is used to estimate the updated advantage alignment policy gradient terms.
    Let k := nb_sub_steps. Then the number of steps generated is O(Tk), where T is
    the maximum trajectory length.
    """

    folder = os.path.split(markov_game.output_path)[0] # get head


    async def run_depth_k_max(markov_game: MarkovGame, k: int):
        """
        Executes k steps in a markov game.
        """
        counter = 0

        # First, take simulation since actions are already set
        terminated = await markov_game.take_simulation_step()
        if terminated:
            markov_game.export()
            return

        # Then, the remaining steps are taken normally
        for _ in range(1, k):
            terminated = await markov_game.step()
            if terminated:
                markov_game.export()
                return
        markov_game.export()
        raise NotImplementedError



    async def run(markov_game: MarkovGame, node: Node):
        """
        Run games whilst branching out with different actions.
        """

        # Take step in main trajectory
        terminated = await markov_game.step()
        if terminated:
            markov_game.export()
            return
        markov_game.set_actions()
        time_step = markov_game.time_step

        # Get alternative branches by generating new unilateral actions
        for agent_id in markov_game.agent_ids:
            for _ in range(nb_alternative_actions):
                mg_branch = markov_game.get_new_branch()
                branch_name = f"alternate_action_of_{agent_id}_time_{time_step}"
                mgb_out_path = os.path.join(folder, branch_name)
                mg_branch.output_path = mgb_out_path
                branch_node = Node(
                    branch_name,
                    payload={
                        "path": mgb_out_path
                        "time_step": time_step,
                        "agent_id": agent_id
                    },
                    parent=node
                )
                # Sample new action
                mg_branch.unset_action_of_agent(agent_id)
                mg_branch.set_action_of_agent(agent_id)
                # Continue trajectory
                asyncio.create_task(run_depth_k_max(markov_game=mg_branch, k=nb_sub_steps))

        await run(markov_game, node=root)

    root = Node("root", payload={"path": markov_game.output_path})
    asyncio.create_task(run(markov_game=markov_game, node=root))

    # TODO: export the Tree object + a pretty print for debugging


async def VinePPORunner(
    markov_game: MarkovGame,
    **kwargs):
    pass
