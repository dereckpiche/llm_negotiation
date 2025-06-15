from typing import Union
from peft import LoraConfig


class RtConfig:
    def __init__(
        self,
        entropy_coeff: float,
        kl_coeff: float,
        gradient_clipping: Union[float, None],
        restrict_tokens: Union[list[str], None],
        mini_batch_size: int,
        use_gradient_checkpointing: bool,
        logging_path: str,
        temperature: float,
        device: str,


        # Value function
        use_gae: bool,
        gae_lambda: float,
        create_fake_bootstrap_value: bool,

        # Regular Rewards
        discount_factor: float,

        # Opponent Shaping 
        use_sum_credits: bool,
        use_advantage_alignment: bool,
        ad_align_normalize_advantages: bool,
        ad_align_force_coop_first_step: bool,
        use_sign_in_ad_align: bool,
        ad_align_clipping: float,
        use_time_regularization_in_ad_align: bool,
        use_variance_regularization_in_ad_align: bool,
        ad_align_beta: float,


        # Regular logging
        log_entropy_gradient_terms: bool = False,
        log_kl_gradient_terms: bool = False,
        log_value_gradient_terms: bool = False,

        # Contextualized logging
        log_ctz_length: int = 30,
        log_ctz_top_k: int = 10,
        log_ctz_next_token: bool = False,
        log_ctz_next_token_score: bool = False,
        log_ctz_next_token_log_prob: bool = False,
        log_ctz_next_token_prob: bool = False,
        log_ctz_top_k_tids: bool = False,
        log_ctz_top_k_probs: bool = False,
        log_ctz_top_slogpi: bool = False,
        log_ctz_entropy: bool = False,
        log_ctz_kl: bool = False,

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
            use_sum_credits:
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
        self.logging_path = logging_path
        self.temperature = temperature
        self.device = device
        self.discount_factor = discount_factor
        self.use_sum_credits = use_sum_credits
        self.create_fake_bootstrap_value = create_fake_bootstrap_value


        self.ad_align_force_coop_first_step = ad_align_force_coop_first_step
        self.ad_align_normalize_advantages = ad_align_normalize_advantages
        self.use_advantage_alignment = use_advantage_alignment
        self.use_variance_regularization_in_ad_align = (
            use_variance_regularization_in_ad_align
        )
        self.use_time_regularization_in_ad_align = use_time_regularization_in_ad_align
        self.ad_align_beta = ad_align_beta

        if use_advantage_alignment or use_sum_credits:
            self.wait_for_opponent_shaping = True
        else:
            self.wait_for_opponent_shaping = False
        self.use_gae = use_gae
        self.gae_lambda = gae_lambda
        self.use_sign_in_ad_align = use_sign_in_ad_align 
        self.ad_align_clipping = ad_align_clipping


        self.log_entropy_gradient_terms = log_entropy_gradient_terms
        self.log_kl_gradient_terms = log_kl_gradient_terms
        self.log_value_gradient_terms = log_value_gradient_terms
        
        self.log_ctz_length = log_ctz_length
        self.log_ctz_top_k = log_ctz_top_k 
        self.log_ctz_next_token=log_ctz_next_token
        self.log_ctz_next_token_score=log_ctz_next_token_score
        self.log_ctz_next_token_log_prob=log_ctz_next_token_log_prob
        self.log_ctz_next_token_prob=log_ctz_next_token_prob
        self.log_ctz_top_k_tids=log_ctz_top_k_tids
        self.log_ctz_top_k_probs=log_ctz_top_k_probs
        self.log_ctz_top_slogpi=log_ctz_top_slogpi
        self.log_ctz_entropy=log_ctz_entropy
        self.log_ctz_kl=log_ctz_kl

