import os
import json
from .dond_statistics_funcs import *
from .dond_return_funcs import *

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

        # Create HTML representation of the game context
        html_games_path = os.path.join(path, player_name, "html_games")
        os.makedirs(html_games_path, exist_ok=True)
        
        # Determine the next available file number for HTML data
        html_files = os.listdir(html_games_path)
        html_numbers = [int(f.split('_')[-1].split('.')[0]) for f in html_files if f.startswith("game_context_")]
        next_html_number = max(html_numbers, default=0) + 1
        html_file = os.path.join(html_games_path, f"game_context_{next_html_number}.html")
        
        # Generate HTML content
        html_content = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Game Context</title>
            <style>
                body { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
                .message { margin-bottom: 20px; padding: 15px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1); }
                .user { background-color: rgba(0, 123, 255, 0.1); }
                .assistant { background-color: rgba(40, 167, 69, 0.1); }
                .role { font-weight: bold; margin-bottom: 5px; }
                .game-info { margin-top: 30px; padding: 15px; border-radius: 10px; background-color: #ffffff; box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1); }
            </style>
        </head>
        <body>
        """

        # Add each message in a sequence of <assistant> and <user> divs
        for message in training_data:
            role = "Intermediary ⚙️" if message["role"] == "user" else "LLM 🤖"
            role_class = "user" if message["role"] == "user" else "assistant"
            html_content += f"""
            <div class="message {role_class}">
                <div class="role">{role}</div>
                <p>{message["content"]}</p>
            </div>
            """

        # Add game information at the bottom
        html_content += """
        <div class="game-info">
            <h3>Game Information</h3>
            <p>Include relevant game details here...</p>
        </div>
        </body>
        </html>
        """
        
        # Write HTML content to file
        with open(html_file, "w") as f:
            f.write(html_content)
