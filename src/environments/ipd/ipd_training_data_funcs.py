from typing import Dict, Any, List, Tuple, Optional, Callable
import numpy as np
from llm_negotiation.src.environments.ipd.ipd_statistics_funcs import calculate_ipd_scores


def generate_training_data_from_raw(
    raw_matches: List[Dict[str, Any]],
    discount_factor: float = 0.99,
    exclude_errors: bool = True,
    score_shaping_function: Optional[Callable] = None,
    score_shaping_function_args: Dict[str, Any] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Generate training data from raw match data for Iterated Prisoner's Dilemma.
    
    Args:
        raw_matches: List of raw match data
        discount_factor: Discount factor for future rewards
        exclude_errors: Whether to exclude trajectories with errors
        score_shaping_function: Function to apply to the scores
        score_shaping_function_args: Arguments for the score shaping function
        
    Returns:
        Training data grouped by policy ID
    """
    if score_shaping_function_args is None:
        score_shaping_function_args = {}
    
    # Group data by policy ID
    training_data = {}
    
    for match in raw_matches:
        # Extract match log and policies
        env_log_info = match.get("env_log_info", {})
        agent_log_infos = match.get("agent_log_infos", {})
        
        # For each agent in the match
        for agent_id, agent_info in agent_log_infos.items():
            # Get policy ID
            policy_id = agent_info.get("policy_id")
            
            if not policy_id:
                continue
            
            # Initialize policy data if not exists
            if policy_id not in training_data:
                training_data[policy_id] = []
            
            # Extract trajectory
            trajectory = match.get("trajectories", {}).get(agent_id, [])
            
            # Skip if there are errors and exclude_errors is True
            has_errors = any(step.get("error") for step in trajectory)
            if has_errors and exclude_errors:
                continue
            
            # Calculate scores for the agent
            raw_scores, total_score = calculate_ipd_scores(match, agent_id)
            
            # Apply discount factor to get returns
            returns = []
            discounted_return = 0
            
            for i in range(len(raw_scores) - 1, -1, -1):
                discounted_return = raw_scores[i] + discount_factor * discounted_return
                returns.insert(0, discounted_return)
            
            # Apply score shaping if provided
            if score_shaping_function:
                returns = score_shaping_function(returns, match, agent_id, **score_shaping_function_args)
            
            # Create training entries
            for i, step in enumerate(trajectory):
                # Skip steps with errors
                if step.get("error"):
                    continue
                
                # Skip steps without policy input
                policy_input = step.get("policy_input")
                if not policy_input:
                    continue
                
                # Skip steps without policy output
                policy_output = step.get("policy_output")
                if not policy_output:
                    continue
                
                # Create training entry
                entry = {
                    "match_id": match.get("match_id"),
                    "agent_id": agent_id,
                    "policy_id": policy_id,
                    "policy_input": policy_input,
                    "policy_output": policy_output,
                    "score": returns[i] if i < len(returns) else 0.0,
                    "meta": {
                        "round": i,
                        "total_score": total_score,
                        "action": step.get("action"),
                    }
                }
                
                training_data[policy_id].append(entry)
    
    return training_data


def calculate_discounted_scores(
    returns: List[float],
    match_info: Dict[str, Any],
    agent_id: str,
    normalize_func: Optional[Callable] = None,
    **kwargs
) -> List[float]:
    """
    Apply additional processing to the calculated returns.
    
    Args:
        returns: List of returns
        match_info: Information about the match
        agent_id: ID of the agent
        normalize_func: Function to normalize the returns
        **kwargs: Additional arguments
        
    Returns:
        Processed returns
    """
    # Apply normalization if provided
    if normalize_func:
        returns = normalize_func(returns, match_info, agent_id, **kwargs)
    
    return returns


def subtract_baseline(
    returns: List[float],
    match_info: Dict[str, Any],
    agent_id: str,
    baseline: float = 0.0,
    **kwargs
) -> List[float]:
    """
    Subtract a constant baseline from the returns.
    
    Args:
        returns: List of returns
        match_info: Information about the match
        agent_id: ID of the agent
        baseline: Baseline value to subtract
        **kwargs: Additional arguments
        
    Returns:
        Returns with baseline subtracted
    """
    return [r - baseline for r in returns]


def subtract_rolling_baseline(
    returns: List[float],
    match_info: Dict[str, Any],
    agent_id: str,
    window_size: int = 10,
    **kwargs
) -> List[float]:
    """
    Subtract a rolling average baseline from the returns.
    
    Args:
        returns: List of returns
        match_info: Information about the match
        agent_id: ID of the agent
        window_size: Size of the rolling window
        **kwargs: Additional arguments
        
    Returns:
        Returns with rolling baseline subtracted
    """
    if not returns:
        return []
    
    processed_returns = []
    for i, r in enumerate(returns):
        # Calculate baseline as average of past returns
        start_idx = max(0, i - window_size)
        if start_idx == i:  # No past returns
            baseline = 0.0
        else:
            baseline = sum(returns[start_idx:i]) / (i - start_idx)
        
        processed_returns.append(r - baseline)
    
    return processed_returns 