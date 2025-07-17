from anytree import Node, RenderTree
from anytree.exporter import DotExporter
import os.path
import asyncio
from mllm.markov_games.markov_game import MarkovGame

async def LinearRunner(
    markov_game: MarkovGame,
    nb_alternative_actions: int = 0,
    nb_sub_steps: int = 1):
    """
    This method generates a trajectory without branching.
    """
    terminated = False
    while not terminated:
        terminated = await markov_game.step()
    markov_game.export()
