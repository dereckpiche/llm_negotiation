
import asyncio

from torch._C import ClassType
from mllm.markov_games.markov_game import MarkovGame
from dataclasses import dataclass
from collections.abc import Callable




async def run_all(
    runner: Callable[[MarkovGame], None],
    markov_games: list[MarkovGame]):
    for mg in markov_games:
        asyncio.create_task(runner(markov_game = mg))
    await asyncio.gather(*asyncio.all_tasks() - {asyncio.current_task()})
