import os
import json
from utils.common_imports import *



def diplomacy_log_match(
        path,
        agents_log_info,
        env_log_info,
        metrics_func=None,
        metrics_func_args=None
        ):
    """
    Logs the Diplomacy game data and generates HTML visualizations using the get_log_info methods.
    
    Args:
        path (str): Base path to save the data.
        agents_log_info (list): List of agent information dictionaries containing the get_log_info results.
        env_log_info (dict): Environment information from its get_log_info method.
        metrics_func (str, optional): Name of the function to calculate metrics.
        metrics_func_args (dict, optional): Arguments for the metrics function.
    """
    # Create directory structure
    os.makedirs(path, exist_ok=True)
    
    # Save the environment log info
    env_log_path = os.path.join(path, "env_log.json")
    with open(env_log_path, "w") as f:
        json.dump(env_log_info, f, indent=4, default=_json_serialize)
    
    # Process each agent's log info
    for agent_log in agents_log_info:
        power_name = agent_log["power_name"]
        
        # Define paths for raw data and statistics subfolders
        power_path = os.path.join(path, power_name)
        raw_data_path = os.path.join(power_path, "raw_data")
        statistics_path = os.path.join(power_path, "statistics")
        
        # Ensure directories exist
        os.makedirs(raw_data_path, exist_ok=True)
        os.makedirs(statistics_path, exist_ok=True)
        
        # Determine the next available file number for raw data
        raw_files = os.listdir(raw_data_path)
        raw_numbers = [int(f.split('_')[-1].split('.')[0]) for f in raw_files if f.startswith("log_")]
        next_raw_number = max(raw_numbers, default=0) + 1
        raw_file = os.path.join(raw_data_path, f"log_{next_raw_number}.json")
        
        # Save agent log info
        with open(raw_file, "w") as f:
            json.dump(agent_log, f, indent=4, default=_json_serialize)
        
        # Log metrics if a metrics function is provided
        if metrics_func:
            metrics_files = os.listdir(statistics_path)
            metrics_numbers = [int(f.split('_')[-1].split('.')[0]) for f in metrics_files if f.startswith("metrics_")]
            next_metrics_number = max(metrics_numbers, default=0) + 1
            metrics_file = os.path.join(statistics_path, f"metrics_{next_metrics_number}.json")
            
            metrics = globals()[metrics_func](agent_log, info, **metrics_func_args)
            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=4)
    
    # Generate the HTML visualization
    html_content = generate_diplomacy_html(agents_log_info, env_log_info)
    
    # Ensure the html directory exists
    html_path = os.path.join(path, "html")
    os.makedirs(html_path, exist_ok=True)
    
    # Determine the next available file number for HTML
    html_files = os.listdir(html_path)
    html_numbers = [int(f.split('_')[-1].split('.')[0]) for f in html_files if f.startswith("game_summary_")]
    next_html_number = max(html_numbers, default=0) + 1
    html_file = os.path.join(html_path, f"game_summary_{next_html_number}.html")
    
    # Save the HTML content to a file
    with open(html_file, "w") as f:
        f.write(html_content)

def generate_diplomacy_html(agent_infos, env_info):
    """
    Generate HTML visualization for a Diplomacy game.
    
    Args:
        agent_infos (list): List of agent information dictionaries from get_log_info.
        env_info (dict): Environment information from get_log_info.
        
    Returns:
        str: HTML content for the game visualization.
    """
    # Extract game information
    game_id = env_info.get("game_id", "Unknown")
    phase = env_info.get("phase", "Unknown")
    map_name = env_info.get("map_name", "standard")
    is_game_done = env_info.get("is_game_done", False)
    outcome = env_info.get("outcome", [])
    
    centers = env_info.get("centers", {})
    units = env_info.get("units", {})
    
    # HTML head and style
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Diplomacy Game {game_id}</title>
        <style>
            body {{
                font-family: 'Arial', sans-serif;
                background-color: #f5f5f5;
                color: #333333;
                margin: 0;
                padding: 20px;
            }}
            .container {{
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                grid-gap: 20px;
                margin-bottom: 30px;
            }}
            .central-info {{
                grid-column: span 3;
                background: #fff;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
                margin-bottom: 20px;
            }}
            .power-column {{
                background: #fff;
                padding: 15px;
                border-radius: 10px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            }}
            .message {{
                margin-bottom: 15px;
                padding: 12px;
                border-radius: 8px;
                box-shadow: 0 1px 4px rgba(0, 0, 0, 0.1);
            }}
            .user {{
                background: rgba(235, 245, 255, 0.8);
                border-left: 4px solid #007bff;
            }}
            .assistant {{
                background: rgba(240, 255, 240, 0.8);
                border-right: 4px solid #28a745;
            }}
            .orders {{
                background: rgba(255, 248, 225, 0.8);
                border-left: 4px solid #ffc107;
            }}
            .role {{
                font-weight: bold;
                margin-bottom: 5px;
                color: #333333;
            }}
            .power-name {{
                text-align: center;
                font-size: 1.4em;
                margin-bottom: 15px;
                color: #000;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 1px;
            }}
            .game-info {{
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                grid-gap: 15px;
            }}
            .info-card {{
                background: #f9f9f9;
                padding: 15px;
                border-radius: 8px;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            }}
            .supply-centers, .units-list {{
                display: flex;
                flex-wrap: wrap;
                justify-content: space-between;
            }}
            .supply-center, .unit {{
                flex: 0 0 30%;
                margin-bottom: 10px;
                padding: 8px;
                background: #f0f0f0;
                border-radius: 5px;
                text-align: center;
            }}
            h2 {{
                border-bottom: 2px solid #eee;
                padding-bottom: 10px;
                margin-top: 0;
            }}
            .outcome {{
                background: #e8f5e9;
                padding: 15px;
                border-radius: 8px;
                margin-top: 15px;
                font-weight: bold;
                text-align: center;
            }}
            .austria {{ border-top: 5px solid #ff5050; }}
            .england {{ border-top: 5px solid #5050ff; }}
            .france {{ border-top: 5px solid #50c0ff; }}
            .germany {{ border-top: 5px solid #808080; }}
            .italy {{ border-top: 5px solid #50ff50; }}
            .russia {{ border-top: 5px solid #ffffff; border: 1px solid #ccc; }}
            .turkey {{ border-top: 5px solid #c0c000; }}
        </style>
    </head>
    <body>
        <div class="central-info">
            <h2>Game Information</h2>
            <div class="game-info">
                <div class="info-card">
                    <h3>Game Details</h3>
                    <p><strong>Game ID:</strong> {game_id}</p>
                    <p><strong>Phase:</strong> {phase}</p>
                    <p><strong>Map:</strong> {map_name}</p>
                    <p><strong>Status:</strong> {status}</p>
                </div>
                <div class="info-card">
                    <h3>Supply Centers</h3>
                    <div class="supply-centers">
    """.format(
        game_id=game_id,
        phase=phase,
        map_name=map_name,
        status="Completed" if is_game_done else "Active"
    )

    # Add supply center information
    for power, power_centers in centers.items():
        html_content += f"""
                        <div class="supply-center">
                            <strong>{power}:</strong> {len(power_centers)}
                        </div>
        """

    html_content += """
                    </div>
                </div>
            </div>
    """

    # Add outcome if game is done
    if is_game_done and outcome:
        winners = outcome[1:] if len(outcome) > 1 else ["Draw"]
        html_content += f"""
            <div class="outcome">
                <h3>Game Outcome</h3>
                <p>Winners: {', '.join(winners)}</p>
            </div>
        """

    html_content += """
        </div>
        <div class="container">
    """

    # Add each power's information
    for agent_log in agent_infos:
        power_name = agent_log["power_name"]
        power_class = power_name.lower()
        orders = agent_log.get("orders", [])
        message_history = agent_log.get("message_history", [])
        
        html_content += f"""
            <div class="power-column {power_class}">
                <div class="power-name">{power_name}</div>
                
                <div class="info-card">
                    <h3>Units</h3>
                    <ul>
        """
        
        # Add units information
        power_units = units.get(power_name, [])
        for unit in power_units:
            html_content += f"<li>{unit}</li>"
        
        html_content += """
                    </ul>
                </div>
                
                <div class="message orders">
                    <div class="role">Final Orders</div>
                    <ul>
        """
        
        # Add orders
        for order in orders:
            html_content += f"<li>{order}</li>"
        
        html_content += """
                    </ul>
                </div>
        """
        
        # Add message history
        for message in message_history:
            if isinstance(message, dict):
                # Skip system messages or handle differently
                if message.get("role") == "system":
                    continue
                
                role = message.get("role", "unknown")
                content = message.get("content", "")
                
                role_class = "user" if role == "user" else "assistant"
                role_display = "Environment" if role == "user" else f"LLM ({power_name})"
                
                # Escape HTML characters in content
                content = content.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
                
                html_content += f"""
                <div class="message {role_class}">
                    <div class="role">{role_display}</div>
                    <p>{content}</p>
                </div>
                """
            elif isinstance(message, str):
                # Simple string messages (may be used in some implementations)
                html_content += f"""
                <div class="message">
                    <p>{message}</p>
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
    
    return html_content

def _json_serialize(obj):
    """
    A helper function to convert non-JSON-serializable objects
    (like OrderResult) into strings or dicts.
    """
    # Check for the specific object types you know are problematic
    if obj.__class__.__name__ == "OrderResult":
        # Return a string representation or a dict
        return str(obj)
    
    # Fallback: attempt to convert anything else to string
    return str(obj)