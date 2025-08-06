import asyncio
from collections.abc import Callable
from dataclasses import dataclass

from torch._C import ClassType

from mllm.markov_games.markov_game import MarkovGame
from mllm.markov_games.rollout_tree import RolloutTreeRootNode


async def run_markov_games(
    runner: Callable[[MarkovGame], RolloutTreeRootNode],
    runner_kwargs: dict,
    output_folder: str,
    markov_games: list[MarkovGame],
) -> list[RolloutTreeRootNode]:
    tasks = []
    for mg in markov_games:
        tasks.append(
            asyncio.create_task(
                runner(markov_game=mg, output_folder=output_folder, **runner_kwargs)
            )
        )
    return await asyncio.gather(*tasks)
