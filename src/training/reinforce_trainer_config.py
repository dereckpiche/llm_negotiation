from typing import Union


class RtConfig:
    def __init__(
        self,
        entropy_coeff: float,
        kl_coeff: float,
        gradient_clipping: Union[float, None],
        restrict_tokens: Union[list[str], None],
        mini_batch_size: int,
        use_gradient_checkpointing: bool,
        top_k_for_logging: int,
        logging_path: str,
        temperature: float,
        discount_factor: float,
        use_sum_rewards: bool,
        use_advantage_alignment: bool,
        use_variance_regularization_in_ad_align: bool,
        use_time_regularization_in_ad_align: bool,
        ad_align_beta: float,
        device="cuda:0",
    ):
        """
        Args:
            entropy_coeff:
                Coefficient of the entropy term in the loss.
            kl_coeff:
                Coefficient of the KL-divergence term in the loss.
            restrict_tokens:
                TODO
            mini_batch_size:
                The number of conversations/trajectories we backpropagate
                through at once. This only affects the GPU usage.
            gradient_clipping:
                Maximum norm of the gradient component before it gets clipped.
            top_k_for_logging:
                For every token generation of the model, the trainer takes
                the k tokens with highest probability mass and token-level entropy, kl, etc.
            discount_factor:
                TODO
            use_sum_rewards:
                TODO
            use_advantage_alignment:
                TODO
            use_variance_regularization_in_ad_align:
                TODO
            use_time_regularization_in_ad_align:
                TODO
            ad_align_beta:
                TODO
            logging_path:
                Path at which metrics are logged.
        """
        self.entropy_coeff = entropy_coeff
        self.kl_coeff = kl_coeff
        self.gradient_clipping = gradient_clipping
        self.restrict_tokens = restrict_tokens
        self.mini_batch_size = mini_batch_size
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.top_k_for_logging = top_k_for_logging
        self.logging_path = logging_path
        self.temperature = temperature
        self.device = device
        self.discount_factor = discount_factor
        self.use_sum_rewards = use_sum_rewards
        self.use_advantage_alignment = use_advantage_alignment
        self.use_variance_regularization_in_ad_align = (
            use_variance_regularization_in_ad_align
        )
        self.use_time_regularization_in_ad_align = use_time_regularization_in_ad_align
        self.ad_align_beta = ad_align_beta
