"""
This file contains the various methods that create the scores from the rewards.
By "scores", we mean the coefficients which me multiply to to log gradients in the REINFORCE/PPO algorithms.

REINFORCE gradient estimator is a sum of these terms:

s(a, s) ∇ log π(a|s)

where s(a, s) is score of the action-state pair.

"""


import numpy as np

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
# Score Utils
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


