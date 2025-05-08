import json
import os
from typing import Any, Dict, List, Optional

from environments.ipd.ipd_statistics_funcs import gather_ipd_statistics


def log_ipd_match(
    path: str,
    agent_infos: List[Dict[str, Any]],
    info: Dict[str, Any],
    metrics_func: Optional[callable] = None,
    metrics_func_args: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Log the IPD match.
    """
    for agent_info in agent_infos:
        agent_id = agent_info["agent_id"]

        # Define paths for raw data and statistics subfolders
        raw_data_path = os.path.join(path, agent_id, "raw_data")
        statistics_path = os.path.join(path, agent_id, "statistics")

        # Log raw match data
        chat_history = agent_info.get("chat_history", [])

        chat_history_with_info = chat_history.copy()
        game_info_message = {
            "role": "system",
            "game_info": info,
            "agent_id": agent_id,
        }
        chat_history_with_info.append(game_info_message)

        with open(raw_file, "w") as f:
            json.dump(chat_history_with_info, f, indent=4)

        # Log metrics if a metrics function is provided
        # Log metrics if a metrics function is provided
        if metrics_func:
            metrics_file = os.path.join(
                statistics_path, f"metrics_mid_{match_id}_gid_{group_id}.json"
            )

            metrics = globals()[metrics_func](agent_info, info, **metrics_func_args)
            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=4)
