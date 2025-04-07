import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd


def generate_training_data_from_raw(
    raw_data_folder,
    training_data_folder,
    exclude_errors=False,
    debug_output=False,
    score_method=None,
    score_method_kwargs=None,
):
    """
    Generates training data from raw match data by calculating scores.

    Args:
        raw_data_folder (str): Path to the folder containing raw match data.
        training_data_folder (str): Path to save the processed training data.
        discount_factor (float): The discount factor to apply to future scores.
        exclude_errors (bool): If True, exclude messages with "is_error" set to True.
        score_normalize_func (callable, optional): Function that takes a list of raw scores and returns a new list
            of shaped scores.
    """

    # Find the score of each round of each game of agent associated with "raw_data_folder"
    round_points_agent, round_points_coagent = get_round_points_arrays(raw_data_folder)
    scores = globals()[score_method](
        round_points_agent, round_points_coagent, **score_method_kwargs
    )

    os.makedirs(training_data_folder, exist_ok=True)
    if debug_output:
        debug_output_folder = os.path.join(
            os.path.dirname(training_data_folder), "training_data_debug"
        )
        os.makedirs(debug_output_folder, exist_ok=True)
        # Export round_points_agent, round_points_coagent, and scores as CSV in debug folder
        pd.DataFrame(round_points_agent).to_csv(
            os.path.join(debug_output_folder, "round_points_agent.csv"), index=False
        )
        pd.DataFrame(round_points_coagent).to_csv(
            os.path.join(debug_output_folder, "round_points_coagent.csv"), index=False
        )
        pd.DataFrame(scores).to_csv(
            os.path.join(debug_output_folder, "scores.csv"), index=False
        )

    # Create training data, giving each action their score
    match_files = [
        f
        for f in os.listdir(raw_data_folder)
        if f.startswith("match_") and f.endswith(".json")
    ]
    match_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    # matches = [json.load(open(os.path.join(raw_data_folder, f), 'r')) for f in match_files]

    if not match_files:
        print(f"No raw data files found in {raw_data_folder}")
        return

    for i, match_file in enumerate(match_files):
        chat_history = json.load(open(os.path.join(raw_data_folder, match_file), "r"))

        if exclude_errors:
            chat_history = [
                msg for msg in chat_history if not msg.get("is_error", False)
            ]

        # Attribute scores to actions
        for message in chat_history:
            if message.get("role") == "assistant":
                round_number = message.get("round_nb")
                message["score"] = scores[i, round_number]

        # Only keep conversation messages, not system info
        chat_history = [
            message for message in chat_history if message.get("role") != "system"
        ]

        # Save file to disk
        training_file = os.path.join(
            training_data_folder, match_file.replace("match_", "training_data_")
        )
        with open(training_file, "w") as f:
            json.dump(chat_history, f, indent=4)

    return


def get_round_points_arrays(raw_data_folder):
    """
    Takes a raw_data_folder path, and generates a round reward array for both agents.
    Each row corresponds to a match.
    """
    match_files = [
        f
        for f in os.listdir(raw_data_folder)
        if f.startswith("match_") and f.endswith(".json")
    ]
    match_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    matches = [
        json.load(open(os.path.join(raw_data_folder, f), "r")) for f in match_files
    ]

    # Get number of rounds
    # TODO (Dereck): we should take role=system instead of indexing with -1s
    game_info = matches[0][-1].get("game_info")
    agent_name = matches[0][-1].get("agent_name")

    nb_games = len(match_files)
    nb_rounds = len(game_info.get("round_agreements_reached"))

    round_points_agent = np.ones(shape=(nb_games, nb_rounds))
    round_points_coagent = np.ones(shape=(nb_games, nb_rounds))

    for i, match in enumerate(matches):
        game_info = match[-1].get("game_info")
        agent_name = match[-1].get("agent_name")
        for round in range(nb_rounds):
            agent_role = game_info.get("round_agent_roles", {})[round].get(agent_name)
            coagent_role = next(
                role
                for role in game_info.get("round_agent_roles", {})[round].values()
                if role != agent_role
            )
            round_points_agent[i, round] = game_info.get("round_points")[round].get(
                agent_role
            )
            round_points_coagent[i, round] = game_info.get("round_points")[round].get(
                coagent_role
            )

    return round_points_agent, round_points_coagent


############################################################
# Score methods
############################################################


def r2g_scores(rewards_agent_1, rewards_agent_2, discount_factor):
    """
    Discounted rewards-to-go scores.
    Most basic RL scores. High variance.
        TODO documentation
    """
    return get_discounted_rewards_to_go(
        rewards_agent_1, discount_factor=discount_factor
    )


def rloo_scores(rewards_agent_1, rewards_agent_2, discount_factor):
    """
    TODO: documentation
    """
    return rewards_to_rloo_advantages(rewards_agent_1, discount_factor=discount_factor)


def rloo_advantage_alignment_scores(
    rewards_agent1, rewards_agent2, discount_factor, beta, regulate_var=False
):
    """
    TODO: documentation
    """
    a1 = rewards_to_rloo_advantages(rewards_agent1, discount_factor=discount_factor)
    a2 = rewards_to_rloo_advantages(rewards_agent2, discount_factor=discount_factor)
    advantage_alignment_scores = advantages_to_aa_scores(
        a1, a2, beta=beta, gamma=discount_factor, regulate_var=regulate_var
    )
    return advantage_alignment_scores


############################################################
# Utils
############################################################


def get_discounted_rewards_to_go(rewards, discount_factor):
    """
    Trajectories assumed to be same length.
    """
    T = rewards.shape[1]
    scores = np.zeros(shape=rewards.shape)
    scores[:, -1] = rewards[:, -1]
    for i in range(T - 2, -1, -1):
        scores[:, i] = rewards[:, i] + discount_factor * scores[:, i + 1]
    return scores


def rewards_to_rloo_advantages(rewards, discount_factor):
    """
    Args:
        rounds_points (np.array): Rows are different matches. Columns are rounds. Components are
        rewards.
    """
    n = rewards.shape[0]
    scores = get_discounted_rewards_to_go(rewards, discount_factor)
    rloo_advantages = scores - (np.sum(scores, axis=0, keepdims=True) - scores) / (
        n - 1
    )
    return rloo_advantages


def advantages_to_aa_scores(a1, a2, beta=1.0, gamma=0.9, regulate_var=False):
    """
    Calculate the advantage alignment scores with vectorization.
    Args:
        a1 (np.ndarray): The first advantage array.
        a2 (np.ndarray): The second advantage array.
        gamma (float, optional): The discount factor. Defaults to 0.9.
        beta (float, optional): The shaping factor. Defaults to 1.0.
    Returns:
        adv_align_terms (np.ndarray): The advantage alignment terms.
    The advantage alignment score is calculated as:
    .. math::
        A^*(s_t, a_t, b_t) = A^1(s_t, a_t, b_t) + \\beta \\gamma \\cdot
        \\left( \\sum_{k < t} \\gamma^{t-k} A^1(s_k, a_k, b_k) \\right)
        A^2(s_t, a_t, b_t)
    Refer to https://arxiv.org/abs/2406.14662
    """
    T = a1.shape[1]
    discounted_a1 = a1 * (gamma * np.ones(shape=(1, T))) ** (-np.arange(0, T, 1))
    discounted_sums_a1 = discounted_a1 @ (np.triu(np.ones((T, T))) - np.identity(T))
    t_discounts = (gamma * np.ones(shape=(1, T))) ** (np.arange(0, T, 1))
    alignment_terms = gamma * t_discounts * discounted_sums_a1 * a2
    if regulate_var == True:
        reg_coef = np.std(a1[:, -1]) / (np.std(alignment_terms[:, -1]) + 1e-10)
    else:
        reg_coef = 1.0
    adv_align_terms = a1 + reg_coef * beta * alignment_terms
    return adv_align_terms


if __name__ == "__main__":
    # Test advantage alignment vectorized method
    beta = 0.7
    gamma = 0.9
    a1 = [3, 9, 4, 15]
    a2 = [14, 12, 2, 47]
    print(f"Element 2 should be {a1[1] + beta * gamma * (gamma**(1-0)*a1[0]) * a2[1]}")
    print(
        f"Element 3 should be {a1[2] + beta * gamma * (gamma**(2-0)*a1[0]+ gamma**(2-1)*a1[1]) * a2[2]}"
    )
    print(
        f"Element 4 should be {a1[3] + beta * gamma * (gamma**(3-0)*a1[0]+ gamma**(3-1)*a1[1]+ gamma**(3-2)*a1[2]) * a2[3]}"
    )
    print(
        advantages_to_aa_scores(np.array([a1]), np.array([a2]), beta=beta, gamma=gamma)
    )
