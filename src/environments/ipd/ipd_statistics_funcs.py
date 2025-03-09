from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from collections import Counter


def gather_ipd_statistics(match_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate statistics for an Iterated Prisoner's Dilemma match.
    
    Args:
        match_info: Information about the match
        
    Returns:
        Dictionary of statistics
    """
    # Extract history from match_info
    history = match_info.get("env_log_info", {}).get("history", [])
    
    if not history:
        return {
            "cooperation_rate": 0.0,
            "mutual_cooperation_rate": 0.0,
            "mutual_defection_rate": 0.0,
            "total_rewards": {},
            "average_round_reward": {},
        }
    
    # Total rewards per agent
    total_rewards = match_info.get("env_log_info", {}).get("total_rewards", {})
    
    # Agent IDs - assuming two agents
    agent_ids = list(total_rewards.keys())
    
    # Extract actions from history
    actions_per_agent = {agent_id: [] for agent_id in agent_ids}
    
    for entry in history:
        for agent_id in agent_ids:
            actions_per_agent[agent_id].append(entry["actions"][agent_id])
    
    # Calculate cooperation rates
    cooperation_rate = {}
    for agent_id, actions in actions_per_agent.items():
        cooperation_count = actions.count("C")
        cooperation_rate[agent_id] = cooperation_count / len(actions) if actions else 0.0
    
    # Calculate mutual cooperation and defection rates
    mutual_cooperation_count = 0
    mutual_defection_count = 0
    
    for entry in history:
        actions = [entry["actions"][agent_id] for agent_id in agent_ids]
        if all(action == "C" for action in actions):
            mutual_cooperation_count += 1
        elif all(action == "D" for action in actions):
            mutual_defection_count += 1
    
    total_rounds = len(history)
    mutual_cooperation_rate = mutual_cooperation_count / total_rounds if total_rounds else 0.0
    mutual_defection_rate = mutual_defection_count / total_rounds if total_rounds else 0.0
    
    # Calculate average reward per round
    average_round_reward = {}
    for agent_id, total_reward in total_rewards.items():
        average_round_reward[agent_id] = total_reward / total_rounds if total_rounds else 0.0
    
    # Prepare statistics
    statistics = {
        "cooperation_rate": cooperation_rate,
        "mutual_cooperation_rate": mutual_cooperation_rate,
        "mutual_defection_rate": mutual_defection_rate,
        "total_rewards": total_rewards,
        "average_round_reward": average_round_reward,
    }
    
    return statistics


def calculate_ipd_scores(match_info: Dict[str, Any], agent_id: str) -> Tuple[List[float], float]:
    """
    Calculate scores for the Iterated Prisoner's Dilemma for a specific agent.
    
    Args:
        match_info: Information about the match
        agent_id: The ID of the agent to calculate scores for
        
    Returns:
        List of scores for each round and total score
    """
    # Extract history from match_info
    history = match_info.get("env_log_info", {}).get("history", [])
    
    if not history:
        return [], 0.0
    
    # Extract rewards from history
    rewards = [entry["rewards"][agent_id] for entry in history]
    total_reward = sum(rewards)
    
    return rewards, total_reward 