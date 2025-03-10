import os
import json
from utils.common_imports import *

def diplomacy_log_match(
        path,
        agent_infos,
        info,
        metrics_func=None,
        metrics_func_args=None
        ):
    """
    Logs the raw conversation data for each Diplomacy power and generates HTML visualizations.
    
    Args:
        path (str): Base path to save the data.
        agent_infos (list): List of agent information dictionaries.
        info (dict): Game information.
        metrics_func (str, optional): Name of the function to calculate metrics.
        metrics_func_args (dict, optional): Arguments for the metrics function.
    """
    # First, perform the normal raw conversation logging for each power
    for agent_info in agent_infos:
        power_name = agent_info["power_name"]

        # Define paths for raw data and statistics subfolders
        raw_data_path = os.path.join(path, power_name, "raw_data")
        statistics_path = os.path.join(path, power_name, "statistics")

        # Ensure directories exist
        os.makedirs(raw_data_path, exist_ok=True)
        os.makedirs(statistics_path, exist_ok=True)

        # Determine the next available file number for raw data
        raw_files = os.listdir(raw_data_path)
        raw_numbers = [int(f.split('_')[-1].split('.')[0]) for f in raw_files if f.startswith("conversation_")]
        next_raw_number = max(raw_numbers, default=0) + 1
        raw_file = os.path.join(raw_data_path, f"conversation_{next_raw_number}.json")

        # Log raw conversation data
        conversation_history = agent_info.get("conversation_history", [])
        
        # Add game info to the conversation history for later processing
        conversation_with_info = conversation_history.copy()
        game_info_message = {
            "role": "system", 
            "game_info": info, 
            "power_name": power_name,
            "current_action": agent_info.get("current_action")
        }
        conversation_with_info.append(game_info_message)
        
        with open(raw_file, "w") as f:
            json.dump(conversation_with_info, f, indent=4)

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
    # Generate HTML content with a layout for the 7 powers
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Diplomacy Game</title>
        <style>
            body {
                font-family: 'Arial', sans-serif;
                background-color: #f5f5f5;
                color: #333333;
                margin: 0;
                padding: 20px;
            }
            .container {
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                grid-gap: 20px;
                margin-bottom: 30px;
            }
            .central-info {
                grid-column: span 3;
                background: #fff;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
                margin-bottom: 20px;
            }
            .power-column {
                background: #fff;
                padding: 15px;
                border-radius: 10px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            }
            .message {
                margin-bottom: 15px;
                padding: 12px;
                border-radius: 8px;
                box-shadow: 0 1px 4px rgba(0, 0, 0, 0.1);
            }
            .user {
                background: rgba(235, 245, 255, 0.8);
                border-left: 4px solid #007bff;
            }
            .assistant {
                background: rgba(240, 255, 240, 0.8);
                border-right: 4px solid #28a745;
            }
            .role {
                font-weight: bold;
                margin-bottom: 5px;
                color: #333333;
            }
            .power-name {
                text-align: center;
                font-size: 1.4em;
                margin-bottom: 15px;
                color: #000;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            .game-info {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                grid-gap: 15px;
            }
            .info-card {
                background: #f9f9f9;
                padding: 15px;
                border-radius: 8px;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            }
            .supply-centers {
                display: flex;
                flex-wrap: wrap;
                justify-content: space-between;
            }
            .supply-center {
                flex: 0 0 30%;
                margin-bottom: 10px;
                padding: 8px;
                background: #f0f0f0;
                border-radius: 5px;
                text-align: center;
            }
            h2 {
                border-bottom: 2px solid #eee;
                padding-bottom: 10px;
                margin-top: 0;
            }
            .austria { border-top: 5px solid #ff5050; }
            .england { border-top: 5px solid #5050ff; }
            .france { border-top: 5px solid #50c0ff; }
            .germany { border-top: 5px solid #808080; }
            .italy { border-top: 5px solid #50ff50; }
            .russia { border-top: 5px solid #ffffff; border: 1px solid #ccc; }
            .turkey { border-top: 5px solid #c0c000; }
        </style>
    </head>
    <body>
        <div class="central-info">
            <h2>Game Information</h2>
            <div class="game-info">
                <div class="info-card">
                    <h3>Current State</h3>
                    <p><strong>Turn:</strong> {current_turn}</p>
                    <p><strong>Season:</strong> {current_season}</p>
                </div>
                <div class="info-card">
                    <h3>Supply Centers</h3>
                    <div class="supply-centers">
    """.format(
        current_turn=info.get("current_turn", "N/A"),
        current_season=info.get("current_season", "N/A")
    )

    # Add supply center information
    supply_centers = info.get("supply_centers", {})
    for power, count in supply_centers.items():
        html_content += f"""
                        <div class="supply-center">
                            <strong>{power}:</strong> {count}
                        </div>
        """

    html_content += """
                    </div>
                </div>
            </div>
        </div>
        <div class="container">
    """

    # Group agents into rows for the 7 powers
    for agent_info in agent_infos:
        power_name = agent_info["power_name"].upper()
        power_class = power_name.lower()
        
        html_content += f"""
            <div class="power-column {power_class}">
                <div class="power-name">{power_name}</div>
        """
        
        # Add conversation history
        conversation_history = agent_info.get("conversation_history", [])
        for message in conversation_history:
            # Skip system messages with game_info
            if message.get("role") == "system" and "game_info" in message:
                continue
                
            role = "Environment" if message["role"] == "user" else f"LLM ({power_name})"
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
        
        # Add current action if available
        if "current_action" in agent_info and agent_info["current_action"]:
            html_content += f"""
            <div class="message current-action">
                <div class="role">Current Action</div>
                <p>{agent_info["current_action"]}</p>
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
