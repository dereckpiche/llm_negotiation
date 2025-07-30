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
    advantages: torch.Tensor, # (B,S)
    beta: float,
    gamma: float,
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
    # TODO: verify
    # Regular alignment terms
    T = advantages.shape[1]
    discounted_advantages = advantages * (gamma * torch.ones(shape=(1, T))) ** (
        -torch.arange(0, T, 1)
    )
    # Identity is for \( k < t \), remove for \( k \leq t \)
    discounted_sums_advantages = discounted_advantages @ (
        torch.triu(torch.ones((T, T))) - torch.identity(T)
    )
    t_discounts = (gamma * torch.ones(shape=(1, T))) ** (torch.arange(0, T, 1))
    adalign_weights = beta * t_discounts * discounted_sums_advantages
    return adalign_weights


def advantages_to_aa_credits(
    a1: torch.Tensor, # (B, S)
    a2: torch.Tensor, # (B, S)
    tally: RtTally
) -> torch.Tensor:
    """
    Calculate the advantage alignment credits with vectorization, as described in https://arxiv.org/abs/2406.14662.
    Applies normalization, sign, clipping, and regularization as specified in config.
    Optionally logs intermediate and final metrics.

    The advantage alignment credit is calculated as:
    \[
        A^*(s_t, a_t, b_t) = A^1(s_t, a_t, b_t) + \\beta \\gamma \\cdot
        \\left( \\sum_{k < t} \\gamma^{t-k} A^1(s_k, a_k, b_k) \\right)
        A^2(s_t, a_t, b_t)
    \]

    Args:
        a1 (torch.Tensor): The first advantage array.
        a2 (torch.Tensor): The second advantage array.

    Returns:
        torch.Tensor: The advantage alignment terms.
    """
    assert not rewards.is_nested, "Input must not be a nested tensor."
    if len(a1.shape) == 1:
        a1 = a1[None, :]
    if len(a2.shape) == 1:
        a2 = a2[None, :]
    assert a1.shape == a2.shape, "Not the same shape"
    T = a1.shape[1]
    a1 = torch.array(a1)
    a2 = torch.array(a2)
    gamma = self.config.discount_factor
    beta = self.config.ad_align_beta

    adalign_weights = self.get_advantage_alignment_weights(
        advantages=a1, beta=beta, gamma=gamma
    )

    self.tally.add_metric(
        path=["raw_advantage_alignment_weights"], metric=adalign_weights
    )

    # Use sign
    if self.config.use_sign_in_ad_align:
        assert beta == 1.0, "beta should be 1.0 when using sign"
        positive_signs = adalign_weights > 0
        negative_signs = adalign_weights < 0
        adalign_weights[positive_signs] = 1
        adalign_weights[negative_signs] = -1
        self.tally.add_metric(
            path=["adalign_weights_ratio_positive_signs"],
            metric=positive_signs.sum() / adalign_weights.size,
        )
        self.tally.add_metric(
            path=["adalign_weights_ratio_negative_signs"],
            metric=negative_signs.sum() / adalign_weights.size,
        )
        # (rest are 0)

        self.tally.add_metric(
            path=["ad_align_weights_after_using_sign"], metric=adalign_weights
        )

    # Use clipping
    if self.config.ad_align_clipping not in [0.0, None]:

        upper_mask = adalign_weights > 1
        lower_mask = adalign_weights < -1

        adalign_weights = torch.clip(
            adalign_weights,
            -self.config.ad_align_clipping,
            self.config.ad_align_clipping,
        )
        clipping_ratio = (torch.sum(upper_mask) + torch.sum(lower_mask)) / upper_mask.size

        self.tally.add_metric(
            path=["ad_align_clipping_ratio"], metric=clipping_ratio
        )

        self.tally.add_metric(
            path=["ad_align_weights_after_clipping"], metric=adalign_weights
        )

    # 1/1+t Regularization
    if self.config.use_time_regularization_in_ad_align:
        t_values = torch.arange(1, T + 1)
        adalign_weights = adalign_weights / t_values
        self.tally.add_metric(
            path=["adalign_weights_after_1_over_t_reg"], metric=adalign_weights
        )

    # Use coop on t=0
    if self.config.ad_align_force_coop_first_step:
        adalign_weights[:, 0] = 1
        self.tally.add_metric(
            path=["adalign_weights_after_force_coop_first_step"],
            metric=adalign_weights,
        )

    opp_shaping_terms = adalign_weights * a2

    self.tally.add_metric(
        path=["ad_align_opp_shaping_terms"], metric=opp_shaping_terms
    )

    # Normalize alignment terms (across same time step)
    if self.config.use_variance_regularization_in_ad_align:
        # TODO: verify
        reg_coef = torch.std(a1[:, -1]) / (torch.std(opp_shaping_terms[:, -1]) + 1e-9)
        opp_shaping_terms *= reg_coef
        self.tally.add_metric(
            path=["opp_shaping_terms_after_var_reg"], metric=opp_shaping_terms
        )

    ad_align_credits = a1 + opp_shaping_terms

    self.tally.add_metric(
        path=["final_advantage_alignment_credits"], metric=ad_align_credits
    )

    self.logger.info(f"\n \n After AdAlign \n  {ram_usage()} \n {vram_usage()}")

    return ad_align_credits.squeeze()
