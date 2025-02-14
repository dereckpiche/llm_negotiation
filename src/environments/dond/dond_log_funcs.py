import os
import json
import re
from .dond_statistics_funcs import *
from .dond_return_funcs import *

def players_logging_and_html(
        path,
        player_infos, 
        info,
        training_data_func,
        training_data_func_args,
        metrics_func,
        metrics_func_args
        ):

    for player_info in player_infos:
        player_name = player_info["player_name"]
        
        # Define paths for training and statistics subfolders
        training_path = os.path.join(path, player_name, "training")
        statistics_path = os.path.join(path, player_name, "statistics")
        
        # Ensure directories exist
        os.makedirs(training_path, exist_ok=True)
        os.makedirs(statistics_path, exist_ok=True)
        
        # Determine the next available file number for training data
        training_files = os.listdir(training_path)
        training_numbers = [int(f.split('_')[-1].split('.')[0]) for f in training_files if f.startswith("training_data_")]
        next_training_number = max(training_numbers, default=0) + 1
        training_file = os.path.join(training_path, f"training_data_{next_training_number}.json")
        
        # Log training data
        training_data = globals()[training_data_func](player_info, info, **training_data_func_args)
        with open(training_file, "w") as f:
            json.dump(training_data, f, indent=4)
        
        # Determine the next available file number for metrics
        metrics_files = os.listdir(statistics_path)
        metrics_numbers = [int(f.split('_')[-1].split('.')[0]) for f in metrics_files if f.startswith("metrics_")]
        next_metrics_number = max(metrics_numbers, default=0) + 1
        metrics_file = os.path.join(statistics_path, f"metrics_{next_metrics_number}.json")
        
        # Log metrics
        metrics = globals()[metrics_func](player_info, info, **metrics_func_args)
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=4)

    # Generate HTML content with a vertical split
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Game Context</title>
        <style>
            body { font-family: '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue', sans-serif; margin: 20px; background-color: #f5f5f5; }
            .container { display: flex; }
            .column { flex: 1; padding: 10px; border-left: 2px solid #ccc; }
            .message { margin-bottom: 20px; padding: 15px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1); }
            .user { background-color: #f0f7ff; }
            .assistant { background-color: #faf9f7; }
            .role { font-weight: bold; margin-bottom: 5px; }
            .game-info { margin-top: 30px; padding: 15px; border-radius: 10px; background-color: #ffffff; box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1); }
            .player-name { text-align: center; font-size: 1.5em; margin-bottom: 20px; }
            .highlight-number { color: #007aff; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="container">
    """

    for player_info in player_infos:
        player_name = player_info["player_name"]
        player_class = "alice" if player_name.lower() == "alice" else "bob"
        html_content += f"""
            <div class="column {player_class}">
                <div class="player-name">{player_name}</div>
        """
        player_data = globals()[training_data_func](player_info, info, **training_data_func_args)
        for message in player_data:
            role = "Intermediary ⚙️" if message["role"] == "user" else f"LLM ({player_name}) 🤖"
            role_class = "user" if message["role"] == "user" else "assistant"
            
            # Escape < and > in the message content
            message_content = message["content"].replace("<", "&lt;").replace(">", "&gt;")
            
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


def independant_players_logging(
        path,
        player_infos, 
        info,
        training_data_func,
        training_data_func_args,
        metrics_func,
        metrics_func_args
        ):
    """
    Logs the training data and metrics independently for each player in a match.
    """
    for player_info in player_infos:
        player_name = player_info["player_name"]
        
        # Define paths for training and statistics subfolders
        training_path = os.path.join(path, player_name, "training")
        statistics_path = os.path.join(path, player_name, "statistics")
        
        # Ensure directories exist
        os.makedirs(training_path, exist_ok=True)
        os.makedirs(statistics_path, exist_ok=True)
        
        # Determine the next available file number for training data
        training_files = os.listdir(training_path)
        training_numbers = [int(f.split('_')[-1].split('.')[0]) for f in training_files if f.startswith("training_data_")]
        next_training_number = max(training_numbers, default=0) + 1
        training_file = os.path.join(training_path, f"training_data_{next_training_number}.json")
        
        # Log training data
        training_data = globals()[training_data_func](player_info, info, **training_data_func_args)
        with open(training_file, "w") as f:
            json.dump(training_data, f, indent=4)
        
        # Determine the next available file number for metrics
        metrics_files = os.listdir(statistics_path)
        metrics_numbers = [int(f.split('_')[-1].split('.')[0]) for f in metrics_files if f.startswith("metrics_")]
        next_metrics_number = max(metrics_numbers, default=0) + 1
        metrics_file = os.path.join(statistics_path, f"metrics_{next_metrics_number}.json")
        
        # Log metrics
        metrics = globals()[metrics_func](player_info, info, **metrics_func_args)
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=4)