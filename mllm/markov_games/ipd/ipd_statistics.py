from __future__ import annotations

from typing import Dict

from mllm.markov_games.rollout_tree import SimulationStepLog


def avg_reward(sl: SimulationStepLog) -> Dict[str, float]:
    # One value per agent at each step
    return {aid: float(v) for aid, v in (sl.rewards or {}).items()}
