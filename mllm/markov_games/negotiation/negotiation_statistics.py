from __future__ import annotations

from typing import Dict

from mllm.markov_games.rollout_tree import SimulationStepLog


def avg_reward(sl: SimulationStepLog) -> Dict[str, float]:
    for aid in sl.rewards.keys():
        if "buffer" in str(aid):
            return None
    # One value per agent at each step
    return {aid: float(v) for aid, v in (sl.rewards or {}).items()}

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


def average_proposal_when_agent_values_item_lower(sl: SimulationStepLog) -> Dict[str, float | None] | None:
    info = sl.info or {}
    if not info or not info.get("is_last_timestep_in_round"):
        return None
    quantities = info.get("quantities") or {}
    splits = info.get("splits") or {}
    values = info.get("values") or {}
    agent_ids = list(sl.rewards.keys())
    for aid in agent_ids:
        if "buffer" in str(aid):
            return None
    agent_0_prop_when_lower = []
    agent_1_prop_when_lower = []
    for item in quantities.keys():
        if float(values[agent_ids[0]][item]) < float(values[agent_ids[1]][item]):
            agent_0_prop_when_lower.append(splits[agent_ids[0]]["items_given_to_self"][item])
        elif float(values[agent_ids[0]][item]) > float(values[agent_ids[1]][item]):
            agent_1_prop_when_lower.append(splits[agent_ids[1]]["items_given_to_self"][item])
    # Compute simple averages; if no qualifying items for an agent leave value as None.
    out: Dict[str, float | None] = {}
    out[str(agent_ids[0])] = (
        sum(agent_0_prop_when_lower) / len(agent_0_prop_when_lower)
        if agent_0_prop_when_lower
        else None
    )
    out[str(agent_ids[1])] = (
        sum(agent_1_prop_when_lower) / len(agent_1_prop_when_lower)
        if agent_1_prop_when_lower
        else None
    )
    return out


def average_proposal_when_agent_values_item_higher(sl: SimulationStepLog) -> Dict[str, float | None] | None:
    info = sl.info or {}
    if not info or not info.get("is_last_timestep_in_round"):
        return None
    quantities = info.get("quantities") or {}
    splits = info.get("splits") or {}
    values = info.get("values") or {}
    agent_ids = list(sl.rewards.keys())
    for aid in agent_ids:
        if "buffer" in str(aid):
            return None
    agent_0_prop_when_lower = []
    agent_1_prop_when_lower = []
    for item in quantities.keys():
        if float(values[agent_ids[0]][item]) > float(values[agent_ids[1]][item]):
            agent_0_prop_when_lower.append(splits[agent_ids[0]]["items_given_to_self"][item])
        elif float(values[agent_ids[0]][item]) < float(values[agent_ids[1]][item]):
            agent_1_prop_when_lower.append(splits[agent_ids[1]]["items_given_to_self"][item])
    # Compute simple averages; if no qualifying items for an agent leave value as None.
    out: Dict[str, float | None] = {}
    out[str(agent_ids[0])] = (
        sum(agent_0_prop_when_lower) / len(agent_0_prop_when_lower)
        if agent_0_prop_when_lower
        else None
    )
    out[str(agent_ids[1])] = (
        sum(agent_1_prop_when_lower) / len(agent_1_prop_when_lower)
        if agent_1_prop_when_lower
        else None
    )
    return out