import hydra
import os
import logging
import time
from omegaconf import OmegaConf
import random
# local imports
from environments.dond_run_matches import run_matches
from environments.dond.dond_game import DondEnv
from models.local_llm import LocalLLM
from models.dummy_local_llm import DummyLocalLLM
from models.server_llm import ServerLLM
from statistics import mean
from utils.plot_curves import plot_curves

from environments.dond.dond_agent import DondAgent
from training.extract_ppo_dataset import extract_ppo_dataset
from training.extract_sft_dataset import extract_sft_dataset
import copy


def get_data(game, agent, agent, n_samples):
    state = game.get_state()
    assert state['has_finalized']
    agent.set_usr_message(state)
    context = agent.get_context()

    contexts = [copy.deepcopy(context) for _ in range(n_samples)]
    responses = agent.prompt(contexts)

    scores = []
    for i in range(n_samples):
        send_to_game, is_finalization, processed_response = agent.process_model_response(responses[i], state)
        if not send_to_game: 
            scores.append(0)
            continue
        else:
            game_copy = copy.deepcopy(game)
            game_copy.step(processed_response, is_finalization)
            state_ = game_copy.get_state()
            if state_['agreement_reached_history'][-1]: scores.append(10)
            else: scores.append(0)
    assert len(scores) == n_samples
    responses = [[{'role':'assistant', 'content':r}] for r in responses]
    return contexts, responses, scores


def run_partial_game(game, agent_0, agent_1, agent):
    current_agent = agent_0
    other_agent = agent_1
    for _ in range(1, 1000):
        # Play one turn at a time
        state = game.get_state()
        send_to_game = False

        while not send_to_game: 
            current_agent.set_usr_message(state)
            context = current_agent.get_context()
            # Generate response using the agent
            response = agent.prompt([context])[0]
            send_to_game, is_finalization, processed_response = current_agent.process_model_response(response, state)
        
        game.step(processed_response, is_finalization)

        if is_finalization:
            return game, other_agent  # Game ends, return the game state and the other agent

        # Swap agents for the next turn
        current_agent, other_agent = other_agent, current_agent


def last_completion(cfg):
    cfg = OmegaConf.to_container(cfg, resolve=True, structured_config_mode='dict')
    NB_TRAINING_STEPS = cfg['N_TRAINING_STEPS']
    NB_SAMPLES = cfg['NB_SAMPLES']

    agent = LocalLLM(**cfg['models']['llama']['init_args'])
    agent_0 = DondAgent(agent_name="agent_a", **cfg['agents']['agent_a']['dond_agent_args'])
    agent_0.game_id = 0
    agent_1 = DondAgent(agent_name="agent_b", **cfg['agents']['agent_b']['dond_agent_args'])
    agent_1.game_id = 1

    dond_game = DondEnv(**cfg['dond_game_args'])
    dond_game, agent = run_partial_game(dond_game, agent_0, agent_1, agent)

    mean_scores = []

    for _ in range(NB_TRAINING_STEPS):
        queries, responses, scores = get_data(dond_game, copy.deepcopy(agent), agent, NB_SAMPLES)
        mean_score = mean(scores)
        mean_scores.append(mean_score)
        #assert mean_score > 0.0
        plot_curves(y_list=[mean_scores], plot_name='MEAN SCORE OVER PPO STEPS')
        agent.train_ppo(queries, responses, scores)

    logging.info('Experiment completed.')





        



