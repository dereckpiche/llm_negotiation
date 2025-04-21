from trl import AutoModelForCausalLMWithValueHead

from training.ppo_train import ppo_train
from training.ppo_train_value_head import ppo_train_value_head
from training.reinforce_training import reinforce_train
from training.rl_convs_processing import paths_to_rl_data
from utils.common_imports import *


def train_main(
    hf_model,
    paths,
    train_func,
    train_func_args,
    train_data_args={},
    output_path=None,
):
    train_output = globals()[train_func](
        hf_model, paths, train_func_args, train_data_args, output_path=output_path
    )
    hf_model.export_current_adapter()
    return train_output


def train_ppo_main(
    hf_model,
    paths,
    train_ppo_args={},
    output_path=None,
):
    contexts_list, scores_list, output_masks_list = paths_to_rl_data(
        hf_model.tokenizer, paths
    )

    if isinstance(hf_model.hf_model, AutoModelForCausalLMWithValueHead):
        ppo_train_value_head(
            model=hf_model.hf_model,
            ref_model=hf_model.hf_model,
            contexts_list=contexts_list,
            scores_list=scores_list,
            output_masks_list=output_masks_list,
            **train_ppo_args
        )
    else:
        ppo_train(
            model=hf_model.hf_model,
            ref_model=hf_model.hf_model,
            contexts_list=contexts_list,
            scores_list=scores_list,
            output_masks_list=output_masks_list,
            **train_ppo_args
        )


def train_reinforce_main(
    hf_model, paths, train_reinforce_args={}, train_data_args={}, output_path=None
):
    contexts_list, scores_list, output_masks_list = paths_to_rl_data(
        hf_model.tokenizer, paths, **train_data_args
    )
    train_output = reinforce_train(
        model=hf_model.hf_model,
        optimizer=hf_model.current_optimizer,
        contexts_list=contexts_list,
        scores_list=scores_list,
        output_masks_list=output_masks_list,
        **train_reinforce_args,
        output_path=output_path,
        tokenizer=hf_model.tokenizer
    )
    return train_output
