
import asyncio

from torch._C import ClassType
from mllm.markov_games.markov_game import MarkovGame
from dataclasses import dataclass
from collections.abc import Callable
from mllm.markov_games.rollout_tree import RolloutTreeRootNode

async def run_markov_games(
    runner: Callable[[MarkovGame], None],
    output_folder: str,
    markov_games: list[MarkovGame]) -> list[RolloutTreeRootNode]:
    for mg in markov_games:
        asyncio.create_task(runner(markov_game = mg, output_folder=output_folder))
    return await asyncio.gather(*asyncio.all_tasks() - {asyncio.current_task()})
