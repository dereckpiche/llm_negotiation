import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd


def generate_training_data_from_raw(
    raw_data_folder,
    training_data_folder,
    exclude_errors=False,
    debug_output=True,
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

    # Determine the maximum number of rounds across all games
    max_rounds = max(
        [
            len(match[-1].get("game_info").get("round_agreements_reached"))
            for match in matches
        ]
    )
    round_points_agent = np.full((len(matches), max_rounds), None)
    round_points_coagent = np.full((len(matches), max_rounds), None)

    for i, match in enumerate(matches):
        game_info = match[-1].get("game_info")
        agent_name = match[-1].get("agent_name")
        nb_rounds = len(game_info.get("round_agreements_reached"))
        for round in range(nb_rounds):
            agent_role = game_info.get("round_agent_roles", {})[round].get(agent_name)
            coagent_role = next(
                (
                    role
                    for role in game_info.get("round_agent_roles", {})[round].values()
                    if role != agent_role
                ),
                None,
            )
            if agent_role and coagent_role:
                round_points_agent[i, round] = game_info.get("round_points")[round].get(
                    agent_role
                )
                round_points_coagent[i, round] = game_info.get("round_points")[
                    round
                ].get(coagent_role)

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


def sum_rloo_scores(rewards_agent_1, rewards_agent_2, discount_factor):
    """
    Sum of discounted rewards-to-go scores.
    """

    r1 = rewards_to_rloo_advantages(rewards_agent_1, discount_factor=discount_factor)
    r2 = rewards_to_rloo_advantages(rewards_agent_2, discount_factor=discount_factor)

    return r1 + r2


def rloo_advantage_alignment_scores(
    rewards_agent_1,
    rewards_agent2,
    discount_factor=1.0,
    beta=1.0,
    regulate_var=False,
    time_decay=False,
):
    """
    TODO: documentation
    """

    a1 = rewards_to_rloo_advantages(rewards_agent_1, discount_factor=discount_factor)
    a2 = rewards_to_rloo_advantages(rewards_agent2, discount_factor=discount_factor)
    advantage_alignment_scores = advantages_to_aa_scores(
        a1,
        a2,
        beta=beta,
        gamma=discount_factor,
        regulate_var=regulate_var,
        time_decay=time_decay,
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
    rewards = np.where(rewards == None, 0, rewards)
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
    if n <= 1:
        return scores
    rloo_advantages = scores - (np.sum(scores, axis=0, keepdims=True) - scores) / (
        n - 1
    )
    return rloo_advantages


def advantages_to_aa_scores(
    a1, a2, beta=1.0, gamma=0.9, regulate_var=False, time_decay=False
):
    """
    Calculate the advantage alignment scores with vectorization.
    Args:
        a1 (np.ndarray): The first advantage array.
        a2 (np.ndarray): The second advantage array.
        gamma (float, optional): The discount factor. Defaults to 0.9.
        beta (float, optional): The shaping factor. Defaults to 1.0.
        regulate_var (bool, optional): Whether to regulate variance. Defaults to False.
        time_decay (bool, optional): Whether to apply 1/t regularization. Defaults to False.
    Returns:
        adv_align_terms (np.ndarray): The advantage alignment terms.
    The advantage alignment score is calculated as:
    .. math::
        A^*(s_t, a_t, b_t) = A^1(s_t, a_t, b_t) + \\beta \\gamma \\cdot
        \\left( \\sum_{k < t} \\gamma^{t-k} A^1(s_k, a_k, b_k) \\right)
        A^2(s_t, a_t, b_t)
    Refer to https://arxiv.org/abs/2406.14662
    """

    # Regular alignment terms
    T = a1.shape[1]
    discounted_a1 = a1 * (gamma * np.ones(shape=(1, T))) ** (-np.arange(0, T, 1))
    discounted_sums_a1 = discounted_a1 @ (np.triu(np.ones((T, T))) - np.identity(T))
    t_discounts = (gamma * np.ones(shape=(1, T))) ** (np.arange(0, T, 1))
    alignment_terms = gamma * t_discounts * discounted_sums_a1 * a2

    # Normalize alignment terms (across same time step)
    if regulate_var:
        reg_coef = np.std(a1[:, -1]) / (np.std(alignment_terms[:, -1]) + 1e-10)
        alignment_terms = reg_coef * alignment_terms

    # 1/t Regularization
    if time_decay:
        t_values = np.arange(1, T + 1)
        alignment_terms = alignment_terms / t_values

    adv_align_terms = a1 + beta * alignment_terms

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

    def readable_advantages_to_aa_scores(
        a1, a2, beta=1.0, gamma=0.9, regulate_var=False, time_decay=False
    ):
        """
        A more readable version of the advantage alignment calculation that shows step-by-step
        how the advantage alignment scores are computed.

        Args:
            a1 (list or np.ndarray): The first advantage array.
            a2 (list or np.ndarray): The second advantage array.
            beta (float, optional): The shaping factor. Defaults to 1.0.
            gamma (float, optional): The discount factor. Defaults to 0.9.
            regulate_var (bool, optional): Whether to regulate variance. Defaults to False.
            time_decay (bool, optional): Whether to apply 1/t regularization. Defaults to False.

        Returns:
            np.ndarray: The advantage alignment scores for each time step.
        """
        # Convert inputs to numpy arrays if they aren't already
        a1 = (
            np.array(a1, dtype=float)
            if not isinstance(a1, np.ndarray)
            else a1.astype(float)
        )
        a2 = (
            np.array(a2, dtype=float)
            if not isinstance(a2, np.ndarray)
            else a2.astype(float)
        )

        # Handle both 1D arrays and 2D arrays (where rows are different matches)
        single_match = a1.ndim == 1
        if single_match:
            a1 = a1.reshape(1, -1)
            a2 = a2.reshape(1, -1)

        T = a1.shape[1]  # Number of time steps
        aa_scores = np.zeros_like(a1)

        # For clarity, process one match at a time
        for match_idx in range(a1.shape[0]):
            match_a1 = a1[match_idx]
            match_a2 = a2[match_idx]

            for t in range(T):
                # Calculate the sum of discounted past advantages
                discounted_sum = 0
                for k in range(t):
                    discounted_sum += (gamma ** (t - k)) * match_a1[k]

                # Calculate the alignment term
                alignment_term = beta * gamma * discounted_sum * match_a2[t]

                # Apply time decay if specified
                if time_decay:
                    alignment_term = alignment_term / (t + 1)

                # Calculate the final score
                aa_scores[match_idx, t] = match_a1[t] + alignment_term

        # If variance regulation is needed, apply it after all calculations
        if regulate_var:
            # Get the last advantage values and alignment terms for each match
            last_idx = T - 1
            a1_last = a1[:, last_idx]
            alignment_terms_last = aa_scores[:, last_idx] - a1[:, last_idx]

            # Calculate standard deviations
            std_a1 = np.std(a1_last)
            std_alignment = np.std(alignment_terms_last) + 1e-10

            # Calculate regulation coefficient
            reg_coef = std_a1 / std_alignment

            # Apply the regulation to all alignment terms
            for match_idx in range(a1.shape[0]):
                for t in range(T):
                    alignment_term = aa_scores[match_idx, t] - a1[match_idx, t]
                    aa_scores[match_idx, t] = a1[match_idx, t] + (
                        alignment_term * reg_coef
                    )

        # If input was 1D, return a 1D array
        if single_match:
            return aa_scores[0]

        return aa_scores

    # Test the readable version and compare with vectorized version
    print("\nComparing readable vs vectorized implementation:")
    vectorized_result = advantages_to_aa_scores(
        np.array([a1]), np.array([a2]), beta=beta, gamma=gamma
    )[0]
    readable_result = readable_advantages_to_aa_scores(a1, a2, beta=beta, gamma=gamma)

    print(f"Readable implementation result: {readable_result}")
    print(f"Vectorized implementation result: {vectorized_result}")
    print(f"Match? {np.allclose(readable_result, vectorized_result)}")

    # Test with a batch of advantages
    print("\nTesting with multiple matches:")
    batch_a1 = np.array([[3, 9, 4, 15], [2, 5, 8, 3]])
    batch_a2 = np.array([[14, 12, 2, 47], [10, 7, 9, 5]])

    batch_vectorized = advantages_to_aa_scores(
        batch_a1, batch_a2, beta=beta, gamma=gamma
    )
    batch_readable = readable_advantages_to_aa_scores(
        batch_a1, batch_a2, beta=beta, gamma=gamma
    )

    print(f"Readable batch result shape: {batch_readable}")
    print(f"Vectorized batch result shape: {batch_vectorized}")
    print(f"Batch match? {np.allclose(batch_readable, batch_vectorized)}")

    # Test rewards_to_rloo_advantages with a simple 3x3 array
    print("\nTesting rewards_to_rloo_advantages with a 3x3 array:")
    test_rewards = np.array(
        [
            [1, 2, 3, 0],  # Match 1 rewards
            [4, 5, 6, 0],  # Match 2 rewards
            [7, 8, 9, 0],  # Match 3 rewards
        ]
    )
    discount_factor = 0.9

    # Calculate expected result manually
    # First calculate discounted rewards to go
    expected_rtg = np.array(
        [
            [1 + 0.9 * 2 + 0.9**2 * 3, 2 + 0.9 * 3, 3],
            [4 + 0.9 * 5 + 0.9**2 * 6, 5 + 0.9 * 6, 6],
            [7 + 0.9 * 8 + 0.9**2 * 9, 8 + 0.9 * 9, 9],
        ]
    )

    # Then calculate leave-one-out advantage
    expected_rloo = np.zeros_like(expected_rtg)
    for i in range(3):
        for j in range(3):
            others_avg = (
                np.sum(expected_rtg[:, j]) - expected_rtg[i, j]
            ) / 2  # n-1 = 2
            expected_rloo[i, j] = expected_rtg[i, j] - others_avg

    rloo_result = rewards_to_rloo_advantages(test_rewards, discount_factor)

    print(f"Test rewards:\n{test_rewards}")
    print(f"Discounted rewards-to-go:\n{expected_rtg}")
    print(f"Expected RLOO advantages:\n{expected_rloo}")
    print(f"Function output:\n{rloo_result}")
    print(f"Match? {np.allclose(expected_rloo, rloo_result)}")
