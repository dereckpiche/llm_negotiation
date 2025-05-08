import json
import os

from utils.common_imports import *
from utils.two_chats_to_html import two_chats_to_html

from .dond_statistics_funcs import *
from .dond_training_data_funcs import *


def dond_log_match(path, agent_infos, info, metrics_func=None, metrics_func_args=None):
    """
    Logs the raw match data for each agent and generates HTML visualizations.

    Args:
        path (str): Base path to save the data.
        agent_infos (list): List of agent information dictionaries.
        info (dict): Game information.
        metrics_func (str, optional): Name of the function to calculate metrics.
        metrics_func_args (dict, optional): Arguments for the metrics function.
    """
    match_id = info["match_id"]
    group_id = info["group_id"]

    # First, perform the normal raw match logging
    for agent_info in agent_infos:
        agent_name = agent_info["agent_name"]

        # Define paths for raw data and statistics subfolders
        raw_data_path = os.path.join(path, agent_name, "raw_data")
        statistics_path = os.path.join(path, agent_name, "statistics")

        # Ensure directories exist
        os.makedirs(raw_data_path, exist_ok=True)
        os.makedirs(statistics_path, exist_ok=True)

        # Determine the next available file number for raw data
        raw_file = os.path.join(
            raw_data_path, f"match_mid_{match_id}_gid_{group_id}.json"
        )

        # Log raw match data
        chat_history = agent_info.get("chat_history", [])

        # Add game info to the chat history for later processing
        chat_history_with_info = chat_history.copy()
        game_info_message = {
            "role": "system",
            "game_info": info,
            "agent_name": agent_name,
        }
        chat_history_with_info.append(game_info_message)

        with open(raw_file, "w") as f:
            json.dump(chat_history_with_info, f, indent=4)

        # Log metrics if a metrics function is provided
        if metrics_func:
            metrics_file = os.path.join(
                statistics_path, f"metrics_mid_{match_id}_gid_{group_id}.json"
            )

            metrics = globals()[metrics_func](agent_info, info, **metrics_func_args)
            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=4)

    # Now generate the HTML visualization
    # Generate HTML content with a vertical split

    html_content = two_chats_to_html(
        agent_infos[0]["chat_history"], agent_infos[1]["chat_history"]
    )

    # Save the HTML content to a file
    with open(html_file, "w") as f:
        f.write(html_content)
