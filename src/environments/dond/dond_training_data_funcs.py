import os
import json
import numpy as np



def generate_training_data_from_raw(raw_data_folder, training_data_folder, discount_factor=0.99, exclude_errors=False, score_shaping_function=None):
    """
    Generates training data from raw conversation data by calculating scores.

    Args:
        raw_data_folder (str): Path to the folder containing raw conversation data.
        training_data_folder (str): Path to save the processed training data.
        discount_factor (float): The discount factor to apply to future scores.
        exclude_errors (bool): If True, exclude messages with "is_error" set to True.
        score_shaping_function (callable, optional): Function that takes a list of raw scores and returns a new list 
            of shaped scores.
    """
    # Create training data directory if it doesn't exist
    os.makedirs(training_data_folder, exist_ok=True)

    # Step 1: Collect all raw data files
    raw_files = [f for f in os.listdir(raw_data_folder) if f.startswith("conversation_") and f.endswith(".json")]

    if not raw_files:
        print(f"No raw data files found in {raw_data_folder}")
        return

    # Step 2: Process each raw data file
    for raw_file in raw_files:
        raw_file_path = os.path.join(raw_data_folder, raw_file)
        with open(raw_file_path, 'r') as f:
            chat_history = json.load(f)

        # Filter out error messages if exclude_errors is True
        if exclude_errors:
            chat_history = [msg for msg in chat_history if not msg.get("is_error", False)]

        # Extract game info
        game_info = next((msg.get("game_info") for msg in chat_history if msg.get("role") == "system"), None)
        if not game_info:
            print(f"No game info found in {raw_file_path}")
            continue

        # Calculate scores for each round
        round_points = game_info.get("round_points", [])
        player_name = chat_history[0].get("player_name", "")
        scores = calculate_discounted_scores(round_points, game_info, player_name, discount_factor, score_shaping_function)

        # Update chat history with scores
        for message in chat_history:
            if message.get("role") != "user":
                round_number = message.get("round_nb")
                if round_number is not None and round_number < len(scores):
                    message["score"] = scores[round_number]

        # Save the processed data
        training_file = os.path.join(training_data_folder, raw_file.replace("conversation_", "training_data_"))
        with open(training_file, 'w') as f:
            json.dump(chat_history, f, indent=4)

def calculate_discounted_scores(round_points, game_info, player_name, discount_factor, score_shaping_function=None):
    """
    Calculates discounted scores for each round.

    Args:
        round_points (list): List of points for each round.
        game_info (dict): Game information including player roles.
        player_name (str): Name of the player.
        discount_factor (float): The discount factor to apply to future scores.
        score_shaping_function (callable, optional): Function that takes a list of raw scores and returns a new list of shaped scores.
    
    Returns:
        list: Discounted scores for each round.
    """
    scores = []
    cumulative_return = 0
    for i in reversed(range(len(round_points))):
        role = game_info['round_player_roles'][i].get(player_name)
        round_value = round_points[i].get(role, 0)
        cumulative_return = round_value + discount_factor * cumulative_return
        scores.insert(0, cumulative_return)
    
    if score_shaping_function:
        scores = score_shaping_function(scores)
    return scores

def set_discounted_scores(player_info, info, discount_factor=0.99, score_shaping_function=None):
    """
    Sets the discounted scores for each message in the conversation.

    Args:
        player_info (dict): Contains the chat history of the player.
        info (dict): Contains the game information including scores.
        discount_factor (float): The discount factor to apply to future scores.
        score_shaping_function (callable, optional): Function that takes a list of raw scores and returns a new list of shaped scores.
    """
    # Extract the chat history and scores from the game info
    chat_history = player_info.get("chat_history", [])
    round_points = info.get("round_points", [])
    player_name = player_info.get("player_name", "")

    # Calculate discounted scores for each round
    scores = []
    cumulative_return = 0
    for i in reversed(range(len(round_points))):
        role = info['round_player_roles'][i].get(player_name)
        round_value = round_points[i].get(role, 0) 
        cumulative_return = round_value + discount_factor * cumulative_return
        scores.insert(0, cumulative_return)

    if score_shaping_function:
        scores = score_shaping_function(scores)

    # Set the discounted scores for each message based on round_number
    for message in chat_history:
        if message["role"] != "user":
            round_number = message["round_nb"]
            if round_number < len(scores):
                message["return"] = scores[round_number]
    return chat_history

def calculate_advantage_alignment_scores(round_points, game_info, player_name, discount_factor=0.99, beta=1, score_shaping_function=None):
    """
    Calculates advantage alignment scores for each round.

    Args:
        round_points (list): List of points for each round.
        game_info (dict): Game information including player roles.
        player_name (str): Name of the player.
        discount_factor (float): The discount factor to apply to future scores.
        beta (float): Weight for the opponent shaping term.
        score_shaping_function (callable, optional): Function that takes a list of raw scores and returns a new list of shaped scores.
    
    Returns:
        list: Advantage alignment scores for each round.
    """
    nb_rounds = len(round_points)
    ordered_points_self = np.zeros(nb_rounds)
    ordered_points_other = np.zeros(nb_rounds)

    for i in range(nb_rounds):
        role = game_info['round_player_roles'][i].get(player_name)
        ordered_points_self[i] = round_points[i].get(role, 0)

        other_role = next(r for r in game_info['round_player_roles'][i].values() if r != role)
        ordered_points_other[i] = round_points[i].get(other_role, 0)

    discounted_scores_self = np.zeros(nb_rounds)
    discounted_scores_other = np.zeros(nb_rounds)

    cum_self = 0
    cum_other = 0

    for i in reversed(range(nb_rounds)):
        cum_self = ordered_points_self[i] + discount_factor * cum_self
        discounted_scores_self[i] = cum_self

        cum_other = ordered_points_other[i] + discount_factor * cum_other
        discounted_scores_other[i] = cum_other

    scores = np.zeros(nb_rounds)

    for t in range(nb_rounds):
        score = discounted_scores_self[t]
        alignment_term = 0
        for k in range(t):
            alignment_term += (discount_factor ** (t - k)) * discounted_scores_self[k]
        score += beta * discount_factor * alignment_term * discounted_scores_other[t]
        scores[t] = score

    scores = scores.tolist()
    if score_shaping_function:
        scores = score_shaping_function(scores)
    return scores

def calculate_sum_scores(round_points, discount_factor=0.99, score_shaping_function=None):
    """
    Calculates the sum of rewards for both players for each round.

    Args:
        round_points (list): List of points for each round.
        discount_factor (float): The discount factor to apply to future scores.
        score_shaping_function (callable, optional): Function that takes a list of raw scores and returns a new list of shaped scores.
    
    Returns:
        list: Sum scores for each round.
    """
    nb_rounds = len(round_points)
    total_rewards = [sum(points.values()) for points in round_points]

    scores = []
    cumulative_return = 0
    for i in reversed(range(nb_rounds)):
        cumulative_return = total_rewards[i] + discount_factor * cumulative_return
        scores.insert(0, cumulative_return)

    if score_shaping_function:
        scores = score_shaping_function(scores)
    return scores

def gaussian_normalization(scores):
    """
    Normalizes scores using Gaussian (z-score) normalization.
    Each score becomes (score - mean) / std.
    """
    s_arr = np.array(scores)
    mu = s_arr.mean()
    sigma = s_arr.std()
    if sigma == 0:
        return scores  # Avoid division by zero; return original scores.
    return ((s_arr - mu) / sigma).tolist()

def subtract_baseline(scores):
    """
    Subtracts the baseline (mean of the scores) from each score.
    Each round score becomes score - mean(scores).
    """
    s_arr = np.array(scores)
    baseline = s_arr.mean()
    return (s_arr - baseline).tolist()

def subtract_rloo_baseline(scores):
    """
    Subtracts the Round Leave-One-Out (RLOO) baseline from each round score.
    For each round i, the baseline is computed as the mean of all scores excluding the current one.
    The transformed score is score_i - ((sum(scores)-score_i)/(n-1)).
    """
    s_arr = np.array(scores)
    n = len(s_arr)
    if n <= 1:
        return scores
    total_sum = s_arr.sum()
    shaped_scores = []
    for i, score in enumerate(s_arr):
        loo_baseline = (total_sum - score) / (n - 1)
        shaped_scores.append(score - loo_baseline)
    return shaped_scores