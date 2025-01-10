

def set_discounted_returns(player_info, info, discount_factor=0.99):
    """
    Sets the discounted returns for each message in the conversation.

    Args:
        player_info (dict): Contains the chat history of the player.
        info (dict): Contains the game information including returns.
        discount_factor (float): The discount factor to apply to future returns.
    """
    # Extract the chat history and returns from the game info
    chat_history = player_info.get("chat_history", [])
    round_points = info.get("round_points", [])
    player_name = player_info.get("player_name", "")

    # Calculate discounted returns for each round
    scores = []
    cumulative_return = 0
    for i in reversed(range(len(round_points))):
        # Use the player's name to get the specific points
        role = info['round_player_roles'][i].get(player_name)
        round_value = round_points[i].get(role, 0) 
        cumulative_return = round_value + discount_factor * cumulative_return
        scores.insert(0, cumulative_return)

    # Set the discounted returns for each message based on round_number
    for message in chat_history:
        if message["role"] != "user":
            round_number = message["round_nb"]
            if round_number < len(scores):
                message["return"] = scores[round_number]
    return chat_history

import numpy as np


def set_discounted_advalign_returns(player_info, info, discount_factor=0.99, beta=1):
    """
    Sets the discounted advantage alignment returns for each message in the conversation.

    Args:
        player_info (dict): Contains the chat history of the player.
        info (dict): Contains the game information including returns.
        discount_factor (float): The discount factor to apply to future returns.
        beta (float): Weight for the opponent shaping term.
    """
    chat_history = player_info.get("chat_history", [])
    round_points = info.get("round_points", [])
    player_name = player_info.get("player_name", "")

    nb_rounds = len(round_points)

    # Calculate rewards for self and opponent
    ordered_points_self = np.zeros(nb_rounds)
    ordered_points_other = np.zeros(nb_rounds)

    for i in range(nb_rounds):
        role = info['round_player_roles'][i].get(player_name)
        ordered_points_self[i] = round_points[i].get(role, 0)

        other_role = next(r for r in info['round_player_roles'][i].values() if r != role)
        ordered_points_other[i] = round_points[i].get(other_role, 0)

    # Calculate discounted returns for self and opponent
    discounted_returns_self = np.zeros(nb_rounds)
    discounted_returns_other = np.zeros(nb_rounds)

    cum_self = 0
    cum_other = 0

    for i in reversed(range(nb_rounds)):
        cum_self = ordered_points_self[i] + discount_factor * cum_self
        discounted_returns_self[i] = cum_self

        cum_other = ordered_points_other[i] + discount_factor * cum_other
        discounted_returns_other[i] = cum_other

    # Calculate alignment scores
    scores = np.zeros(nb_rounds)

    for t in range(nb_rounds):
        # Standard RL term
        score = discounted_returns_self[t]

        # Opponent alignment term
        alignment_term = 0
        for k in range(t):
            alignment_term += (discount_factor ** (t - k)) * discounted_returns_self[k]
        score += beta * discount_factor * alignment_term * discounted_returns_other[t]

        scores[t] = score

    # Update chat history with scores
    for message in chat_history:
        if message["role"] != "user":
            round_number = message["round_nb"]
            if round_number < len(scores):
                message["return"] = scores[round_number]

    return chat_history

    