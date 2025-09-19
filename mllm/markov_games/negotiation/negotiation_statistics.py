from __future__ import annotations

from typing import Dict

from mllm.markov_games.rollout_tree import SimulationStepLog


def avg_reward(sl: SimulationStepLog) -> Dict[str, float]:
    for aid in sl.rewards.keys():
        if "buffer" in str(aid):
            return None
    # One value per agent at each step
    return {aid: float(v) for aid, v in (sl.rewards or {}).items()}


def split_greed(sl: SimulationStepLog) -> Dict[str, float] | None:
    info = sl.info or {}
    if not info or not info.get("is_last_timestep_in_round"):
        return None
    quantities = info.get("quantities") or {}
    splits = info.get("splits") or {}
    out: Dict[str, float] = {}
    item_keys = list(quantities.keys())
    for aid in sl.rewards.keys():
        if "buffer" in str(aid):
            return None
    # get total items proposed for each category
    totals = {item: 0 for item in item_keys}
    for _, split in splits.items():
        for item in item_keys:
            totals[item] += float(split["items_given_to_self"][item])
    # compute greed per agent
    for aid, split in splits.items():
        agent_greed = []
        for item in item_keys:
            if item in split["items_given_to_self"]:
                denom = max(float(quantities[item]), float(totals[item]))
                agent_greed.append(float(split["items_given_to_self"][item]) / denom)
        out[str(aid)] = sum(agent_greed) / len(agent_greed) if agent_greed else 0.0
    return out


def split_efficiency(sl: SimulationStepLog) -> Dict[str, float] | None:
    info = sl.info or {}
    if not info or not info.get("is_last_timestep_in_round"):
        return None
    quantities = info.get("quantities") or {}
    values = info.get("values") or {}
    if not values or not quantities:
        return None
    item_keys = list(values.values())[0].keys()
    max_vals, max_quantities = [], []
    for item in item_keys:
        max_val = max(float(agent_vals[item]) for agent_vals in values.values())
        max_vals.append(max_val)
        max_quantities.append(quantities[item])
    for aid in sl.rewards.keys():
        if "buffer" in str(aid):
            return None
    achieved = sum(float(v) for v in sl.rewards.values())
    max_reward = sum(d * v for d, v in zip(max_quantities, max_vals))
    # Efficiency is a global metric; emit same value for a special key "all"
    return {"all_agents": achieved / max_reward}
