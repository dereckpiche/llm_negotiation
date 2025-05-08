import json
import os
from typing import Any, Dict, List, Optional

from src.environments.ipd.ipd_statistics import gather_ipd_statistics
from src.environments.two_chats_to_html import two_chats_to_html

def log_ipd_match(
    path: str,
    agent_infos: List[Dict[str, Any]],
    info: Dict[str, Any],
    ) -> Dict[str, Any]:
    """
    Log the IPD match.
    Args:
        path: The path to the folder where the match will be logged.
        agent_infos: A list of dictionaries containing the agent information.
        info: A dictionary containing the match information.
    Returns:
        A dictionary containing the match information.
    """

    
    match_id = info["match_id"]
    group_id = info["group_id"]

    for agent_info in agent_infos:

        agent_id = agent_info["agent_id"]

        # Define paths for raw data and statistics subfolders
        raw_data_path = os.path.join(path, agent_id, "raw_data")
        raw_data_file = os.path.join(raw_data_path, f"match_mid_{match_id}_gid_{group_id}.json")
        statistics_path = os.path.join(path, agent_id, "statistics")

        # Ensure directories exist
        os.makedirs(raw_data_path, exist_ok=True)
        os.makedirs(statistics_path, exist_ok=True)

        # Log raw match data
        chat_history = agent_info.get("chat_history", [])

        chat_history_with_info = chat_history.copy()
        game_info_message = {
            "role": "system",
            "game_info": info,
            "agent_id": agent_id,
        }
        chat_history_with_info.append(game_info_message)

        with open(raw_data_file, "w") as f:
            json.dump(chat_history_with_info, f, indent=4)

    html_content = two_chats_to_html(
        agent_infos[0]["chat_history"], agent_infos[1]["chat_history"]
    )

    html_file = os.path.join(path, f"match_mid_{match_id}_gid_{group_id}.html")
    with open(html_file, "w") as f:
        f.write(html_content)
