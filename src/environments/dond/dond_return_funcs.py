

def set_discounted_scores(player_info, info, discount_factor=0.99):
    """
    Sets the discounted scores for each message in the conversation.

    Args:
        player_info (dict): Contains the chat history of the player.
        info (dict): Contains the game information including scores.
        discount_factor (float): The discount factor to apply to future scores.
    """
    # Extract the chat history and scores from the game info
    chat_history = player_info.get("chat_history", [])
    round_points = info.get("round_points", [])
    player_name = player_info.get("player_name", "")

    # Calculate discounted scores for each round
    scores = []
    cumulative_return = 0
    for i in reversed(range(len(round_points))):
        # Use the player's name to get the specific points
        role = info['round_player_roles'][i].get(player_name)
        round_value = round_points[i].get(role, 0) 
        cumulative_return = round_value + discount_factor * cumulative_return
        scores.insert(0, cumulative_return)

    # Set the discounted scores for each message based on round_number
    for message in chat_history:
        if message["role"] != "user":
            round_number = message["round_nb"]
            if round_number < len(scores):
                message["return"] = scores[round_number]
    return chat_history

import numpy as np


def set_discounted_advalign_scores(player_info, info, discount_factor=0.99, beta=1):
    """
    Sets the discounted advantage alignment scores for each message in the conversation.

    Args:
        player_info (dict): Contains the chat history of the player.
        info (dict): Contains the game information including scores.
        discount_factor (float): The discount factor to apply to future scores.
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

    # Calculate discounted scores for self and opponent
    discounted_scores_self = np.zeros(nb_rounds)
    discounted_scores_other = np.zeros(nb_rounds)

    cum_self = 0
    cum_other = 0

    for i in reversed(range(nb_rounds)):
        cum_self = ordered_points_self[i] + discount_factor * cum_self
        discounted_scores_self[i] = cum_self

        cum_other = ordered_points_other[i] + discount_factor * cum_other
        discounted_scores_other[i] = cum_other

    # Calculate alignment scores
    scores = np.zeros(nb_rounds)

    for t in range(nb_rounds):
        # Standard RL term
        score = discounted_scores_self[t]

        # Opponent alignment term
        alignment_term = 0
        for k in range(t):
            alignment_term += (discount_factor ** (t - k)) * discounted_scores_self[k]
        score += beta * discount_factor * alignment_term * discounted_scores_other[t]

        scores[t] = score

    # Update chat history with scores
    for message in chat_history:
        if message["role"] != "user":
            round_number = message["round_nb"]
            if round_number < len(scores):
                message["return"] = scores[round_number]

    return chat_history

def set_discounted_sum_rewards_scores(player_info, info, discount_factor=0.99):
    """
    Sets the discounted total scores (sum of rewards for both players) for each message in the conversation.
    
    For each round, the total reward is calculated as:
        total_reward_t = r^i_t + r^j_t
    and the discounted return is computed as:
        return_t = total_reward_t + discount_factor * total_reward_(t+1) + discount_factor^2 * total_reward_(t+2) + ...
    
    Args:
        player_info (dict): Contains the chat history of the player.
        info (dict): Contains the game information including round points.
        discount_factor (float): The discount factor to apply to future scores.

    Returns:
        list: Chat history updated with discounted total scores for non-user messages.
    """
    chat_history = player_info.get("chat_history", [])
    round_points = info.get("round_points", [])
    nb_rounds = len(round_points)
    
    # Calculate total rewards per round: r^i_t + r^j_t
    total_rewards = []
    for i in range(nb_rounds):
        # Sum the rewards from all players for round i.
        total_reward = sum(round_points[i].values())
        total_rewards.append(total_reward)
    
    # Calculate discounted total scores for each round
    scores = []
    cumulative_return = 0
    for i in reversed(range(nb_rounds)):
        cumulative_return = total_rewards[i] + discount_factor * cumulative_return
        scores.insert(0, cumulative_return)
    
    # Update chat history with the discounted total scores based on round_number
    for message in chat_history:
        if message.get("role") != "user":
            round_number = message.get("round_nb")
            if round_number is not None and round_number < nb_rounds:
                message["return"] = scores[round_number]
    
    return chat_history

    