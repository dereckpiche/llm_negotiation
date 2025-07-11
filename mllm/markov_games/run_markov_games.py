
import asyncio
from mllm.markov_games.markov_game import MarkovGame


async def run_all(runner: function, markov_games: list[MarkovGame]):
    for mg in markov_games:
        asyncio.create_task(runner(markov_game = mg))
    await asyncio.gather(*asyncio.all_tasks() - {asyncio.current_task()})
