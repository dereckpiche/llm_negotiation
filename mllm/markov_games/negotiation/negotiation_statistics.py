from __future__ import annotations

from typing import Dict

from mllm.markov_games.rollout_tree import SimulationStepLog


def split_greed(sl: SimulationStepLog) -> Dict[str, float] | None:
    info = sl.info or {}
    if not info or not info.get("is_last_timestep_in_round"):
        return None
    quantities = info.get("quantities") or {}
    denom = float(quantities.get("coins", 1.0)) or 1.0
    splits = info.get("splits") or {}
    out: Dict[str, float] = {}
    for aid in sl.rewards.keys():
        if "buffer" in str(aid):
            return None
    for aid, split in splits.items():
        try:
            out[str(aid)] = float(split["items_given_to_self"]["coins"]) / denom
        except Exception:
            continue
    return out


def split_efficiency(sl: SimulationStepLog) -> Dict[str, float] | None:
    info = sl.info or {}
    if not info or not info.get("is_last_timestep_in_round"):
        return None
    quantities = info.get("quantities") or {}
    denom = float(quantities.get("coins", 1.0)) or 1.0
    values = info.get("values") or {}
    if not values:
        return None
    try:
        max_val = max(float(v) for v in values.values())
    except Exception:
        return None
    if not denom or not max_val:
        return None
    for aid in sl.rewards.keys():
        if "buffer" in str(aid):
            return None
    achieved = sum(float(v) for v in (sl.rewards or {}).values())
    max_reward = denom * max_val
    if not max_reward:
        return None
    # Efficiency is a global metric; emit same value for a special key "all"
    return {"all": achieved / max_reward}
