from mllm.training.tally import RtTally
import torch

NestedTensor = torch.Tensor
def apply_method_on_nested(method, nested_tensor_inputs: list[torch.Tensor]) -> NestedTensor:
    """
    Applies in for loop. When possible, this should be replaced with a padded version.
    """
    B = nested_tensor_inputs[0].shape[0]
    res = []
    for b in range(B):
        res.append(method(*[t.unbind()[b].unsqueeze(0) for t in nested_tensor_inputs]))
    return torch.nested.nested_tensor(res, layout=torch.jagged)


def get_discounted_returns(
    rewards: torch.Tensor, # (B, T)
    discount_factor: float
) -> torch.Tensor :
    """
    Computes Monte Carlo discounted returns for a sequence of rewards.

    Args:
        rewards (torch.Tensor): Array of rewards for each timestep.

    Returns:
        torch.Tensor: Array of discounted returns.
    """
    # TODO: verify, some changes were made here

    assert rewards.dim() == 2, "Wrong dimensions."


    # # Apply method in for loop instead
    # if rewards.is_nested:
    #     f = lambda rewards : get_discounted_returns(rewards, discount_factor)
    #     return apply_method_on_nested(f, [rewards])

    T = rewards.shape[1]
    discounted_returns = torch.zeros_like(rewards)
    accumulator = 0.0
    for t in reversed(range(T)):
        accumulator = rewards[:, t] + discount_factor * accumulator
        discounted_returns[:, t] = accumulator
    return discounted_returns

def get_generalized_advantage_estimates(
    rewards: torch.Tensor, # (B, T)
    value_estimates: torch.Tensor, # (B, T+1)
    discount_factor: float,
    lambda_coef: float,
) -> torch.Tensor:
    """
    Computes Generalized Advantage Estimates (GAE) for a sequence of rewards and value estimates.
    See https://arxiv.org/pdf/1506.02438 for details.


    Returns:
        torch.Tensor: Array of GAE values.
    """
    assert rewards.dim() == value_estimates.dim() == 2, "Wrong dimensions."

    # # Apply method in for loop instead
    # if rewards.is_nested:
    #     f = lambda rewards, value_estimates : get_generalized_advantage_estimates(rewards, value_estimates, discount_factor, lambda_coef)
    #     return apply_method_on_nested(f, [rewards, value_estimates])

    assert rewards.shape[0] == value_estimates.shape[0], f"Got shapes {rewards.shape} and {value_estimates.shape}."
    assert rewards.shape[1] == value_estimates.shape[1]-1, f"Got shapes {rewards.shape} and {value_estimates.shape}."

    T = rewards.shape[1]
    tds = rewards + lambda_coef * value_estimates[:, 1:] - value_estimates[:, :-1]
    gaes = torch.zeros_like(tds)
    acc = 0.0
    for t in reversed(range(T)):
        acc = tds[:, t] + lambda_coef * discount_factor * acc
        gaes[:, t] = acc
    return gaes

def get_advantage_alignment_weights(
    advantages: torch.Tensor, # (B, T)
    exclude_k_equals_t: bool,
) -> torch.Tensor:
    """
    The advantage alignment credit is calculated as

    \[
        A^*(s_t, a_t, b_t) = A^1(s_t, a_t, b_t) + \beta \cdot
        \left( \sum_{k < t} \gamma^{t-k} A^1(s_k, a_k, b_k) \right)
        A^2(s_t, a_t, b_t)
    \]

    Here, the weights are defined as \( \beta \cdot
        \left( \sum_{k < t} \gamma^{t-k} A^1(s_k, a_k, b_k) \)
    """
    T = advantages.shape[1]

    if exclude_k_equals_t:
        sub = torch.identity(T)
    else:
        sub = torch.zeros((T, T))

        
    # Identity is for \( k < t \), remove for \( k \leq t \)
    ad_align_weights = advantages @ (
        torch.triu(torch.ones((T, T))) - sub
    )
    # t_discounts = (gamma * torch.ones(shape=(1, T))) ** (torch.arange(0, T, 1))
    # adalign_weights = beta * t_discounts * discounted_sums_advantages
    return ad_align_weights


def get_advantage_alignment_credits(
    a1: torch.Tensor, # (B, S)
    a1_alternative: torch.Tensor, # (B, A, S)
    a2: torch.Tensor, # (B, S)
    discount_factor: float,
    exclude_k_equals_t: bool,
    beta: float,
    gamma: float,
    use_sign_in_ad_align: bool,
    ad_align_clipping: float,
    use_time_regularization_in_ad_align: bool,
    ad_align_force_coop_first_step: bool,
    use_variance_regularization_in_ad_align: bool,
    tally: RtTally,
    ) -> torch.Tensor:
    """
    Calculate the advantage alignment credits with vectorization, as described in https://arxiv.org/abs/2406.14662.

    Recall that the advantage opponent shaping term of the AdAlign policy gradient is:
    \[
        \beta \mathbb{E}_{\substack{
        \tau \sim \text{Pr}_{\mu}^{\pi^1, \pi^2} \\
        a_t' \sim \pi^1(\cdot \mid s_t)
        }} 
        \left[\sum_{t=0}^\infty  \gamma^{t}\left( \sum_{k\leq t} A^1(s_k,a^{\prime}_k,b_k) \right) A^{2}(s_t,a_t, b_t)\nabla_{\theta^1}\text{log } \pi^1(a_t|s_t) \right]
    \]

    This method computes the following:
    \[
        Credit(s_t, a_t, b_t) = \gamma^t \left[ A^1(s_t, a_t, b_t) + \beta \left( \sum_{k\leq t} A^1(s_k,a^{\prime}_k,b_k) \right) A^{2}(s_t,a_t, b_t) \right]
    \]

    Args:
        a1: Advantages of the main trajectories for the current agent.
        a1_alternative: Advantages of the alternative trajectories for the current agent.
        a2: Advantages of the main trajectories for the other agent.
        discount_factor: Discount factor for the advantage alignment.
        beta: Beta parameter for the advantage alignment.
        gamma: Gamma parameter for the advantage alignment.
        use_sign_in_ad_align: Whether to use sign in the advantage alignment.

    Returns:
        torch.Tensor: The advantage alignment credits.
    """
    assert a1.dim() == a2.dim() == 2, "Advantages must be of shape (B, S)"
    assert a1_alternative.dim() == 3, "Alternative advantages must be of shape (B, A, S)"
    assert a1.shape == a2.shape, "Not the same shape"
    assert a1.shape == a1_alternative.shape, "Not the same shape"
    B, A, T = a1_alternative.shape

    a1_alternative = a1_alternative.mean(dim=1)

    adalign_weights = get_advantage_alignment_weights(
        advantages=a1_alternative, include_k_equals_t=include_k_equals_t
    )

    tally.add_metric(
        path=["raw_advantage_alignment_weights"], metric=adalign_weights
    )

    # Use sign
    if use_sign_in_ad_align:
        assert beta == 1.0, "beta should be 1.0 when using sign"
        positive_signs = adalign_weights > 0
        negative_signs = adalign_weights < 0
        adalign_weights[positive_signs] = 1
        adalign_weights[negative_signs] = -1
        tally.add_metric(
            path=["adalign_weights_ratio_positive_signs"],
            metric=positive_signs.sum() / adalign_weights.size,
        )
        tally.add_metric(
            path=["adalign_weights_ratio_negative_signs"],
            metric=negative_signs.sum() / adalign_weights.size,
        )
        # (rest are 0)

        tally.add_metric(
            path=["ad_align_weights_after_using_sign"], metric=adalign_weights
        )

    ###################
    # Process weights
    ###################

    # Use clipping
    if ad_align_clipping not in [0.0, None]:

        upper_mask = adalign_weights > 1
        lower_mask = adalign_weights < -1

        adalign_weights = torch.clip(
            adalign_weights,
            -ad_align_clipping,
            ad_align_clipping,
        )
        clipping_ratio = (torch.sum(upper_mask) + torch.sum(lower_mask)) / upper_mask.size

        tally.add_metric(
            path=["ad_align_clipping_ratio"], metric=clipping_ratio
        )

        tally.add_metric(
            path=["ad_align_weights_after_clipping"], metric=adalign_weights
        )

    # 1/1+t Regularization
    if use_time_regularization_in_ad_align:
        t_values = torch.arange(1, T + 1)
        adalign_weights = adalign_weights / t_values
        tally.add_metric(
            path=["adalign_weights_after_1_over_t_reg"], metric=adalign_weights
        )

    # Use coop on t=0
    if ad_align_force_coop_first_step:
        adalign_weights[:, 0] = 1
        tally.add_metric(
            path=["adalign_weights_after_force_coop_first_step"],
            metric=adalign_weights,
        )

    # # Normalize alignment terms (across same time step)
    # if use_variance_regularization_in_ad_align:
    #     # TODO: verify
    #     reg_coef = torch.std(a1[:, -1]) / (torch.std(opp_shaping_terms[:, -1]) + 1e-9)
    #     opp_shaping_terms *= reg_coef
    #     tally.add_metric(
    #         path=["opp_shaping_terms_after_var_reg"], metric=opp_shaping_terms
    #     )

    ####################################
    # Compose elements together 
    ####################################

    opp_shaping_terms = beta * adalign_weights * a2

    tally.add_metric(
        path=["ad_align_opp_shaping_terms"], metric=opp_shaping_terms
    )

    t_discounts = (gamma * torch.ones(shape=(1, T))) ** (torch.arange(0, T, 1))
    credits = t_discounts * (a1 + opp_shaping_terms)

    tally.add_metric(
        path=["final_advantage_alignment_credits"], metric=credits
    )

    return credits
