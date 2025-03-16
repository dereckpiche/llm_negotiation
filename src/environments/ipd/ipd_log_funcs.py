from typing import Dict, Any, List, Optional
import json
import os
from environments.ipd.ipd_statistics_funcs import gather_ipd_statistics

def ipd_log_raw_conversations(
    log_dir: str,
    match_infos: List[Dict[str, Any]],
    env_info: Dict[str, Any],
    metrics_func: Optional[callable] = None,
    metrics_func_args: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Log raw conversations from IPD matches.
    
    Args:
        log_dir: Directory to save logs
        match_infos: List of match information dictionaries
        env_info: Information about the environment
        metrics_func: Function to calculate metrics or string name of function
        metrics_func_args: Arguments for the metrics function
        
    Returns:
        Metrics calculated from the matches
    """
    if metrics_func_args is None:
        metrics_func_args = {}
    
    os.makedirs(log_dir, exist_ok=True)
    
    all_metrics = {}
    
    for match_idx, match_info in enumerate(match_infos):
        # Save the raw match data to a JSON file
        with open(os.path.join(log_dir, f"match_{match_idx}.json"), "w") as f:
            json.dump(match_info, f)
        
        # Calculate metrics if a metrics function is provided
        if metrics_func is not None:
            # If metrics_func is a string, resolve it to the actual function
            if isinstance(metrics_func, str):
                if metrics_func == "gather_ipd_statistics":
                    from environments.ipd.ipd_statistics_funcs import gather_ipd_statistics
                    match_metrics_data = gather_ipd_statistics(match_info, env_info)
                else:
                    raise ValueError(f"Unknown metrics function: {metrics_func}")
            else:
                # If it's a callable, use it directly
                match_metrics_data = metrics_func(match_info, env_info, **metrics_func_args)
                
            all_metrics[f"match_{match_idx}"] = match_metrics_data
    
    # Save all metrics to a JSON file
    if all_metrics:
        with open(os.path.join(log_dir, "metrics.json"), "w") as f:
            json.dump(all_metrics, f)
    
    return all_metrics 