from typing import Dict, Any, List, Optional
import json
import os
from llm_negotiation.src.environments.ipd.ipd_statistics_funcs import gather_ipd_statistics


def log_raw_conversations(
    match_infos: List[Dict[str, Any]],
    log_dir: str,
    metrics_func: Optional[callable] = None,
    metrics_func_args: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Log raw conversations from IPD matches.
    
    Args:
        match_infos: List of match information dictionaries
        log_dir: Directory to save logs
        metrics_func: Function to calculate metrics
        metrics_func_args: Arguments for the metrics function
        
    Returns:
        Metrics calculated from the matches
    """
    if metrics_func_args is None:
        metrics_func_args = {}
    
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Save raw match data
    raw_data_path = os.path.join(log_dir, "raw_matches.json")
    with open(raw_data_path, "w") as f:
        json.dump(match_infos, f, indent=2)
    
    # Process and log individual matches
    match_metrics = []
    summary_metrics = {
        "total_matches": len(match_infos),
        "cooperation_rate": {},
        "mutual_cooperation_rate": 0.0,
        "mutual_defection_rate": 0.0,
        "average_rewards": {},
    }
    
    for i, match_info in enumerate(match_infos):
        # Calculate metrics for the match
        if metrics_func:
            match_metrics_data = metrics_func(match_info, **metrics_func_args)
        else:
            match_metrics_data = gather_ipd_statistics(match_info)
        
        match_metrics.append(match_metrics_data)
        
        # Log individual match
        match_path = os.path.join(log_dir, f"match_{i}.json")
        with open(match_path, "w") as f:
            json.dump({
                "match_info": match_info,
                "metrics": match_metrics_data
            }, f, indent=2)
        
        # Update summary metrics
        for agent_id, coop_rate in match_metrics_data.get("cooperation_rate", {}).items():
            if agent_id not in summary_metrics["cooperation_rate"]:
                summary_metrics["cooperation_rate"][agent_id] = []
            summary_metrics["cooperation_rate"][agent_id].append(coop_rate)
        
        summary_metrics["mutual_cooperation_rate"] += match_metrics_data.get("mutual_cooperation_rate", 0.0)
        summary_metrics["mutual_defection_rate"] += match_metrics_data.get("mutual_defection_rate", 0.0)
        
        for agent_id, reward in match_metrics_data.get("total_rewards", {}).items():
            if agent_id not in summary_metrics["average_rewards"]:
                summary_metrics["average_rewards"][agent_id] = []
            summary_metrics["average_rewards"][agent_id].append(reward)
    
    # Calculate averages for summary metrics
    total_matches = len(match_infos)
    if total_matches > 0:
        summary_metrics["mutual_cooperation_rate"] /= total_matches
        summary_metrics["mutual_defection_rate"] /= total_matches
        
        for agent_id, rewards in summary_metrics["average_rewards"].items():
            summary_metrics["average_rewards"][agent_id] = sum(rewards) / len(rewards) if rewards else 0.0
        
        for agent_id, coop_rates in summary_metrics["cooperation_rate"].items():
            summary_metrics["cooperation_rate"][agent_id] = sum(coop_rates) / len(coop_rates) if coop_rates else 0.0
    
    # Save summary metrics
    summary_path = os.path.join(log_dir, "summary_metrics.json")
    with open(summary_path, "w") as f:
        json.dump(summary_metrics, f, indent=2)
    
    return summary_metrics 