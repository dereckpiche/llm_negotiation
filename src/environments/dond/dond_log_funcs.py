import json
import os

from utils.common_imports import *

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
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Game Context</title>
        <style>
            body {
                font-family: 'Arial', sans-serif;
                background-color: #ffffff;
                color: #333333;
                margin: 0;
                padding: 20px;
            }
            .container {
                display: flex;
                justify-content: space-between;
                background: linear-gradient(135deg, rgba(255, 255, 255, 0.95), rgba(245, 248, 255, 0.95));
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
                position: relative;
                min-height: 100vh;
            }
            .column {
                flex: 1;
                margin: 10px;
                padding: 20px;
                border-radius: 10px;
                background: rgba(255, 255, 255, 0.9);
                box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
                border: 2px solid #cccccc;
                transition: transform 0.2s;
                position: relative;
                display: flex;
                flex-direction: column;
                min-height: calc(100vh - 60px);
            }
            .alice {
                border-left: 5px solid #3c78d8; /* Blue for Alice */
            }
            .bob {
                border-left: 5px solid #f9cb9c; /* Light orange for Bob */
            }
            .message {
                margin-bottom: 15px;
                padding: 12px;
                border-radius: 10px;
                box-shadow: 0 1px 4px rgba(0, 0, 0, 0.1);
                position: relative;
            }
            .user {
                background: rgba(230, 255, 230, 0.8); /* Light green for intermediary */
                border-left: 4px solid #34a853;
            }
            .assistant.alice-message {
                background: rgba(225, 240, 255, 0.8); /* Light blue for Alice */
                border-right: 4px solid #3c78d8;
            }
            .assistant.bob-message {
                background: rgba(255, 245, 230, 0.8); /* Very light orange for Bob */
                border-right: 4px solid #f9cb9c;
            }
            .role {
                font-weight: bold;
                margin-bottom: 5px;
                color: #333333;
            }
            .round-divider {
                text-align: center;
                font-weight: bold;
                color: #666;
                margin: 20px 0;
                border-top: 2px dashed #ccc;
                padding-top: 5px;
                position: relative;
                background: #f5f5f5;
                border-radius: 5px;
                z-index: 5;
            }
            .end-round-divider {
                text-align: center;
                font-weight: bold;
                color: #666;
                margin: 20px 0;
                border-top: 2px solid #f44336;
                padding-top: 5px;
                position: relative;
                background: #ffebee;
                border-radius: 5px;
                z-index: 5;
            }
            .game-info {
                margin-top: 20px;
                padding: 15px;
                border-radius: 10px;
                background: rgba(255, 255, 255, 0.9);
                box-shadow: 0 1px 4px rgba(0, 0, 0, 0.1);
            }
            .agent-name {
                text-align: center;
                font-size: 1.4em;
                margin-bottom: 15px;
                color: black;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            .highlight-number {
                color: #0056b3;
                font-weight: bold;
            }
            .user .role {
                text-align: left;
            }
            .assistant .role {
                text-align: right;
            }
            .message-timestamp {
                font-size: 0.8em;
                color: #999;
                text-align: right;
                margin-top: 5px;
            }
            .alice .agent-name {
                color: #3c78d8;
            }
            .bob .agent-name {
                color: #e69138;
            }
            .central-timeline {
                position: absolute;
                left: 50%;
                top: 0;
                bottom: 0;
                width: 2px;
                background-color: #ccc;
                transform: translateX(-50%);
                z-index: 0;
            }
            .spacer {
                flex-grow: 1;
                min-height: 20px;
            }
            .message-placeholder {
                height: 100px; /* Fixed height for consistent spacing */
                margin-bottom: 15px;
                visibility: hidden;
            }
            .round-container {
                width: 100%;
                display: flex;
                flex-direction: column;
            }
            /* Add wrapper for round dividers for better alignment */
            .round-section {
                width: 100%;
                display: flex;
                flex-direction: column;
                flex-grow: 1;
            }
            /* Ensure end-of-round dividers are always at the bottom of their section */
            .round-section .end-round-divider {
                margin-top: auto;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="central-timeline"></div>
    """

    # Extract all messages from all agents with their global order
    all_messages = []
    for agent_info in agent_infos:
        agent_name = agent_info["agent_name"]
        chat_history = agent_info.get("chat_history", [])

        for message in chat_history:
            # Skip system messages with game_info
            if message.get("role") == "system" and "game_info" in message:
                continue

            # Add agent name and global order
            message["agent_name"] = agent_name
            # Calculate a reasonable global order based on round and turn
            round_nb = message.get("round_nb", 0)
            turn_order = message.get("turn_order", 0)
            message["global_order"] = (round_nb * 100) + turn_order

            all_messages.append(message)

    # Sort all messages by their global order
    all_messages.sort(key=lambda x: x.get("global_order", 0))

    # Pre-process to find all round numbers
    all_rounds = set()
    for message in all_messages:
        round_nb = message.get("round_nb")
        if round_nb is not None:
            all_rounds.add(round_nb)

    # Sort rounds for consistent display
    all_rounds = sorted(list(all_rounds))
    if not all_rounds and all_messages:
        all_rounds = [0]  # Default to round 0 if no rounds found but messages exist

    # Group messages by agent and round
    messages_by_agent = {agent_info["agent_name"]: [] for agent_info in agent_infos}

    for message in all_messages:
        agent_name = message["agent_name"]
        messages_by_agent[agent_name].append(message)

    # Group messages by round for each agent
    rounds_by_agent = {agent_name: {} for agent_name in messages_by_agent.keys()}

    for agent_name, messages in messages_by_agent.items():
        for message in messages:
            round_nb = message.get("round_nb", 0)
            if round_nb not in rounds_by_agent[agent_name]:
                rounds_by_agent[agent_name][round_nb] = []
            rounds_by_agent[agent_name][round_nb].append(message)

    # Calculate the exact maximum message count per round
    max_messages_per_round = {}
    for round_nb in all_rounds:
        max_count = 0
        for agent_name in messages_by_agent.keys():
            max_count = max(
                max_count, len(rounds_by_agent[agent_name].get(round_nb, []))
            )
        max_messages_per_round[round_nb] = max_count

    # Render agent columns
    for agent_info in agent_infos:
        agent_name = agent_info["agent_name"]
        agent_class = "alice" if agent_name.lower() == "alice" else "bob"

        html_content += f"""
            <div class="column {agent_class}">
                <div class="agent-name">{agent_name}</div>
        """

        # Process rounds in order
        for round_nb in all_rounds:
            # Add round start divider - always show it, even for round 0
            html_content += f"""
            <div class="round-section">
                <div class="round-divider">Round {round_nb}</div>
                <div class="round-container">
            """

            # Get messages for this round and agent
            round_messages = rounds_by_agent[agent_name].get(round_nb, [])

            # Sort messages by global order
            round_messages.sort(key=lambda x: x.get("global_order", 0))

            # Calculate how many messages we need to display for this round
            max_msgs = max_messages_per_round[round_nb]
            messages_to_show = len(round_messages)
            placeholders_needed = max_msgs - messages_to_show

            # Render actual messages
            for message in round_messages:
                role = (
                    "Intermediary ⚙️"
                    if message["role"] == "user"
                    else f"{agent_name} 🤖"
                )
                role_class = "user" if message["role"] == "user" else "assistant"

                # Add agent-specific class for assistant messages
                if role_class == "assistant":
                    role_class += f" {agent_class}-message"

                # Escape < and > in the message content
                message_content = (
                    message["content"].replace("<", "&lt;").replace(">", "&gt;")
                )
                message_content = message_content.replace("\n", "<br>")

                html_content += f"""
                <div class="message {role_class}">
                    <div class="role">{role}</div>
                    <p>{message_content}</p>
                    <div class="message-timestamp">Round {round_nb}</div>
                </div>
                """

            # Add placeholder divs if needed for alignment
            for _ in range(placeholders_needed):
                html_content += """
                <div class="message-placeholder"></div>
                """

            # Close the round container
            html_content += """
                </div>
            """

            # Add end of round divider
            if round_nb < max(all_rounds):
                html_content += f"""
                <div class="end-round-divider">End of Round {round_nb}</div>
            </div>
                """
            else:
                html_content += """
            </div>
                """

        # End of game divider after final messages
        if all_rounds:
            html_content += f"""
            <div class="end-round-divider">End of Game</div>
            """

        html_content += """
            </div>
        """

    html_content += """
        </div>
    </body>
    </html>
    """

    # Ensure the html directory exists
    html_path = os.path.join(path, "html")
    os.makedirs(html_path, exist_ok=True)

    html_file = os.path.join(
        html_path, f"game_context_mid_{match_id}_gid_{group_id}.html"
    )

    # Save the HTML content to a file
    with open(html_file, "w") as f:
        f.write(html_content)
