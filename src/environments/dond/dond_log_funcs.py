import os
import json
from utils.common_imports import *
from .dond_statistics_funcs import *
from .dond_training_data_funcs import *

def dond_log_match(
        path,
        agent_infos,
        info,
        metrics_func=None,
        metrics_func_args=None
        ):
    """
    Logs the raw match data for each agent and generates HTML visualizations.
    
    Args:
        path (str): Base path to save the data.
        agent_infos (list): List of agent information dictionaries.
        info (dict): Game information.
        metrics_func (str, optional): Name of the function to calculate metrics.
        metrics_func_args (dict, optional): Arguments for the metrics function.
    """
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
        raw_files = os.listdir(raw_data_path)
        raw_numbers = [int(f.split('_')[-1].split('.')[0]) for f in raw_files if f.startswith("match_")]
        next_raw_number = max(raw_numbers, default=0) + 1
        raw_file = os.path.join(raw_data_path, f"match_{next_raw_number}.json")

        # Log raw match data
        chat_history = agent_info.get("chat_history", [])
        
        # Add game info to the chat history for later processing
        chat_history_with_info = chat_history.copy()
        game_info_message = {"role": "system", "game_info": info, "agent_name": agent_name}
        chat_history_with_info.append(game_info_message)
        
        with open(raw_file, "w") as f:
            json.dump(chat_history_with_info, f, indent=4)

        # Log metrics if a metrics function is provided
        if metrics_func:
            metrics_files = os.listdir(statistics_path)
            metrics_numbers = [int(f.split('_')[-1].split('.')[0]) for f in metrics_files if f.startswith("metrics_")]
            next_metrics_number = max(metrics_numbers, default=0) + 1
            metrics_file = os.path.join(statistics_path, f"metrics_{next_metrics_number}.json")

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
            }
            .container {
                display: flex;
                justify-content: space-between;
                background: linear-gradient(135deg, rgba(255, 255, 255, 0.95), rgba(245, 248, 255, 0.95));
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
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
            }
            .message {
                margin-bottom: 15px;
                padding: 12px;
                border-radius: 10px;
                box-shadow: 0 1px 4px rgba(0, 0, 0, 0.1);
            }
            .user {
                background: rgba(235, 245, 255, 0.8); /* Very light blue */
                border-left: 4px solid #007bff;
            }
            .assistant {
                background: rgba(240, 255, 240, 0.8); /* Very light green */
                border-right: 4px solid #28a745;
            }
            .role {
                font-weight: bold;
                margin-bottom: 5px;
                color: #333333;
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
        </style>
    </head>
    <body>
        <div class="container">
    """

    for agent_info in agent_infos:
        agent_name = agent_info["agent_name"]
        agent_class = "alice" if agent_name.lower() == "alice" else "bob"
        html_content += f"""
            <div class="column {agent_class}">
                <div class="agent-name">{agent_name}</div>
        """
        # Use chat_history directly instead of extracting via training_data_func
        chat_history = agent_info.get("chat_history", [])
        for message in chat_history:
            # Skip system messages with game_info
            if message.get("role") == "system" and "game_info" in message:
                continue
                
            role = "Intermediary ⚙️" if message["role"] == "user" else f"LLM ({agent_name}) 🤖"
            role_class = "user" if message["role"] == "user" else "assistant"

            # Escape < and > in the message content
            message_content = message["content"].replace("<", "&lt;").replace(">", "&gt;")
            message_content = message_content.replace("\n", "<br>")

            html_content += f"""
            <div class="message {role_class}">
                <div class="role">{role}</div>
                <p>{message_content}</p>
            </div>
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

    # Determine the next available file number for HTML
    html_files = os.listdir(html_path)
    html_numbers = [int(f.split('_')[-1].split('.')[0]) for f in html_files if f.startswith("game_context_")]
    next_html_number = max(html_numbers, default=0) + 1
    html_file = os.path.join(html_path, f"game_context_{next_html_number}.html")

    # Save the HTML content to a file
    with open(html_file, "w") as f:
        f.write(html_content)

        
