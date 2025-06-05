"""
This file contains the methods used to convert raw data into training data
for the Negotiation game. (Also called Deal or No Deal).
"""

import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

from environments.scores import *


def dond_generate_training_data_from_raw(
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
    round_points_agent, round_points_coagent = dond_get_round_points_arrays(
        raw_data_folder
    )

    (
        scores,
        advantages,
        round_points_agent_ord,
        round_points_coagent_ord,
    ) = dond_get_scores(
        round_points_agent=round_points_agent,
        round_points_coagent=round_points_coagent,
        score_method=score_method,
        score_method_kwargs=score_method_kwargs,
    )

    os.makedirs(training_data_folder, exist_ok=True)
    if debug_output:
        debug_output_folder = os.path.join(
            os.path.dirname(training_data_folder), "training_data_debug"
        )
        os.makedirs(debug_output_folder, exist_ok=True)
        # Export round_points_agent, round_points_coagent, and scores as CSV in debug folder
        pd.DataFrame(round_points_agent_ord).to_csv(
            os.path.join(debug_output_folder, "round_points_agent.csv"), index=False
        )
        pd.DataFrame(round_points_coagent_ord).to_csv(
            os.path.join(debug_output_folder, "round_points_coagent.csv"), index=False
        )
        pd.DataFrame(scores).to_csv(
            os.path.join(debug_output_folder, "scores.csv"), index=False
        )
        pd.DataFrame(advantages).to_csv(
            os.path.join(debug_output_folder, "advantages.csv"), index=False
        )

    # Create training data, giving each action their score
    match_files = [
        f
        for f in os.listdir(raw_data_folder)
        if f.startswith("match_") and f.endswith(".json")
    ]
    match_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

    if not match_files:
        print(f"No raw data files found in {raw_data_folder}")
        return

    for i, match_file in enumerate(match_files):
        chat_history = json.load(open(os.path.join(raw_data_folder, match_file), "r"))
        game_info = get_system_msg(chat_history).get("game_info")
        match_id = game_info["match_id"]
        group_id = game_info["group_id"]

        if exclude_errors:
            chat_history = [
                msg for msg in chat_history if not msg.get("is_error", False)
            ]

        # Attribute scores to actions
        for message in chat_history:
            if message.get("role") == "assistant":
                round_number = message.get("round_nb")
                # Attribute the score corresponding to correct minibatch / group, match and round number.
                message["score"] = scores[group_id][match_id][round_number]

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


def get_system_msg(match):
    system_msg = next((d for d in match if d.get("role") == "system"), None)
    return system_msg


def dond_get_round_points_arrays(raw_data_folder):
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
            len(get_system_msg(match).get("game_info").get("round_agreements_reached"))
            for match in matches
        ]
    )
    round_points_agent = np.full((len(matches), max_rounds), None)
    round_points_coagent = np.full((len(matches), max_rounds), None)

    # Initialize nested dictionaries to store RETURNS for each match within each minibatch / group
    # Format: {minibatch_id: {match_id: [returns]}}
    group_to_round_points_agent = defaultdict(lambda: defaultdict(list))
    group_to_round_points_coagent = defaultdict(lambda: defaultdict(list))

    for i, match in enumerate(matches):
        system_msg = get_system_msg(match)
        game_info = system_msg.get("game_info")
        agent_name = system_msg.get("agent_name")
        nb_rounds = len(game_info.get("round_agreements_reached"))
        match_id = game_info["match_id"]
        group_id = game_info["group_id"]

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

        group_to_round_points_agent[group_id][match_id] = round_points_agent[i]
        group_to_round_points_coagent[group_id][match_id] = round_points_coagent[i]

    return group_to_round_points_agent, group_to_round_points_coagent


def dond_get_scores(
    round_points_agent, round_points_coagent, score_method, score_method_kwargs
):
    # Initialize nested dictionaries to store SCORES for each match within each minibatch group
    # Format: {minibatch_id: {match_id: [scores]}}
    scores = defaultdict(lambda: defaultdict(list))
    advantages = defaultdict(lambda: defaultdict(list))
    round_points_agent_ord = defaultdict(lambda: defaultdict(list))
    round_points_coagent_ord = defaultdict(lambda: defaultdict(list))

    # Loop in all the groups and compute the scores only for the minibatch / group
    for group_id in round_points_agent:
        # Extract all match ids in a minibatch / group
        match_ids = sorted(round_points_agent[group_id].keys())

        # Create arrays that store returns for all matches in a minibatch / group
        agent_points = np.array(
            [round_points_agent[group_id][match_id] for match_id in match_ids]
        )
        coagent_points = np.array(
            [round_points_coagent[group_id][match_id] for match_id in match_ids]
        )

        # Compute the scores from returns of a given minibatch / group
        score, advantage = globals()[score_method](
            agent_points, coagent_points, **score_method_kwargs
        )
        if advantage is None:
            advantage = np.zeros_like(score)
        # Store the scores for each match_id in a minibatch / group
        for match_id, agent_score, agent_advantage in zip(match_ids, score, advantage):
            scores[group_id][match_id] = agent_score
            advantages[group_id][match_id] = agent_advantage
        for match_id, x, y in zip(match_ids, agent_points, coagent_points):
            round_points_agent_ord[group_id][match_id] = x
            round_points_coagent_ord[group_id][match_id] = y

    return scores, advantages, round_points_agent_ord, round_points_coagent_ord
