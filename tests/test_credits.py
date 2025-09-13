import numpy as np
import torch

from mllm.training.credit_methods import get_advantage_alignment_credits


def advantage_align(
    a1: torch.Tensor, a2: torch.Tensor, gamma: float, beta: float
) -> torch.Tensor:
    """
    Compute the advantage alignment term A* in PyTorch.

    Args:
        a1: torch.Tensor of shape (B, T), agent's advantages A1_t
        a2: torch.Tensor of shape (B, T), opponent's advantages A2_t
        gamma: discount factor (float)
        beta: coefficient for alignment term (float)

    Returns:
        A_star: torch.Tensor of shape (B, T)
    """
    B, T = a1.shape
    cum = torch.zeros_like(a1)

    # recurrence: cum[:, t] = gamma * cum[:, t-1] + a1[:, t-1]
    for t in range(1, T):
        cum[:, t] = gamma * cum[:, t - 1] + a1[:, t - 1]

    A_star = a1 + beta * gamma * cum * a2
    return A_star


beta = 1.0
gamma = 0.9
a1 = torch.tensor([[1.0, 4.0, 0.0], [2.0, 1.0, 0.0], [8.0, 7.0, 1.0]])
a2 = torch.tensor([[6.0, 5, 0.0], [5.0, 1.0, 3.0], [3.0, 0.0, 1.0]])

ad_align, _ = get_advantage_alignment_credits(
    a1=a1,
    a2=a2,
    a1_alternative=None,
    exclude_k_equals_t=True,
    beta=beta,
    gamma=gamma,
    use_old_ad_align=True,
)
ad_align_star = advantage_align(a1, a2, gamma, beta)

print(ad_align)
print(ad_align_star)
