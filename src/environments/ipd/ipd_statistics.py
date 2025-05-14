"""
Iterated Prisoner's Dilemma (IPD) Game Statistics

This module provides a modular approach to calculating statistics for Iterated
Prisoner's Dilemma game data. It follows a similar pattern to the Deal or No Deal
statistics module with specialized statistics for IPD.

Key features:
- Modular stat calculation functions that each return (stat_name, stat_value)
- A flexible stat aggregation function (get_ipd_iteration_stats)
- Formatting utilities for organizing the stats
- File handling utilities for loading and saving statistics

Usage:
    # Load data
    data = get_raw_data_files(path, agent_id)

    # Select which stat functions to use
    stat_functions = [
        calc_cooperation_rate,
        calc_total_points,
        calc_mutual_cooperation_rate
    ]

    # Calculate statistics
    stats = get_ipd_iteration_stats(data, stat_functions)
"""

import argparse
import re

from utils.common_imports import *
from utils.leafstats import *

############################################################
# Modular Statistics functions
############################################################


def calc_cooperation_rate(data, format_options=None):
    """
    Calculates the percentage of rounds where the agent cooperated.

    Args:
        data (list): Raw data files containing game information
        format_options (list, optional): Formatting options

    Returns:
        tuple: ("cooperation_rate", percentage value)
    """
    total_cooperations = 0
    total_rounds = 0

    for game_data in data:
        game_info = game_data[-1].get("game_info", {})
        agent_id = game_data[-1].get("agent_id")

        if not agent_id or not game_info:
            continue

        actions = game_info.get("actions", [])
        for round_actions in actions:
            agent_action = round_actions.get(agent_id)
            if agent_action:
                total_rounds += 1
                if agent_action in ["C", "<Cooperate>", "<A>"]:
                    total_cooperations += 1

    if total_rounds > 0:
        cooperation_rate = (total_cooperations / total_rounds) * 100
    else:
        cooperation_rate = 0

    return "cooperation_rate", cooperation_rate


def get_number_of_rounds(data, format_options=None):
    game_data = data[0]
    game_info = game_data[-1].get("game_info", {})
    nb_rounds = game_info.get("round_nb")
    return "number_of_rounds", nb_rounds


def calc_defection_rate(data, format_options=None):
    """
    Calculates the percentage of rounds where the agent defected.

    Args:
        data (list): Raw data files containing game information
        format_options (list, optional): Formatting options

    Returns:
        tuple: ("defection_rate", percentage value)
    """
    total_defections = 0
    total_rounds = 0

    for game_data in data:
        game_info = game_data[-1].get("game_info", {})
        agent_id = game_data[-1].get("agent_id")

        if not agent_id or not game_info:
            continue

        actions = game_info.get("actions", [])
        for round_actions in actions:
            agent_action = round_actions.get(agent_id)
            if agent_action:
                total_rounds += 1
                if agent_action == ["D", "<Defect>", "<B>"]:
                    total_defections += 1

    if total_rounds > 0:
        defection_rate = (total_defections / total_rounds) * 100
    else:
        defection_rate = 0

    return "defection_rate", defection_rate


def calc_total_points(data, format_options=None):
    """
    Calculates the total points for the agent and opponent across all games.

    Args:
        data (list): Raw data files containing game information
        format_options (list, optional): Formatting options

    Returns:
        tuple: ("total_points", {"agent": points, "opponent": points})
    """
    agent_points = 0
    opponent_points = 0

    for game_data in data:
        game_info = game_data[-1].get("game_info", {})
        agent_id = game_data[-1].get("agent_id")

        if not agent_id or not game_info:
            continue

        # Find opponent ID
        agent_ids = game_info.get("agent_ids", [])
        opponent_id = next((id for id in agent_ids if id != agent_id), None)

        rewards = game_info.get("rewards", [])
        for round_rewards in rewards:
            agent_points += round_rewards.get(agent_id, 0)
            opponent_points += round_rewards.get(opponent_id, 0)

    return "total_points", {"agent": agent_points, "opponent": opponent_points}


def calc_mutual_cooperation_rate(data, format_options=None):
    """
    Calculates the percentage of rounds where both agents cooperated.

    Args:
        data (list): Raw data files containing game information
        format_options (list, optional): Formatting options

    Returns:
        tuple: ("mutual_cooperation_rate", percentage value)
    """
    mutual_cooperations = 0
    total_rounds = 0

    for game_data in data:
        game_info = game_data[-1].get("game_info", {})
        agent_id = game_data[-1].get("agent_id")

        if not agent_id or not game_info:
            continue

        # Find opponent ID
        agent_ids = game_info.get("agent_ids", [])
        opponent_id = next((id for id in agent_ids if id != agent_id), None)

        actions = game_info.get("actions", [])
        for round_actions in actions:
            agent_action = round_actions.get(agent_id)
            opponent_action = round_actions.get(opponent_id)
            if agent_action and opponent_action:
                total_rounds += 1
                if agent_action in ["C", "<Cooperate>", "<A>"] and opponent_action in [
                    "C",
                    "<Cooperate>",
                    "<A>",
                ]:
                    mutual_cooperations += 1

    if total_rounds > 0:
        mutual_cooperation_rate = (mutual_cooperations / total_rounds) * 100
    else:
        mutual_cooperation_rate = 0

    return "mutual_cooperation_rate", mutual_cooperation_rate


def calc_mutual_defection_rate(data, format_options=None):
    """
    Calculates the percentage of rounds where both agents defected.

    Args:
        data (list): Raw data files containing game information
        format_options (list, optional): Formatting options

    Returns:
        tuple: ("mutual_defection_rate", percentage value)
    """
    mutual_defections = 0
    total_rounds = 0

    for game_data in data:
        game_info = game_data[-1].get("game_info", {})
        agent_id = game_data[-1].get("agent_id")

        if not agent_id or not game_info:
            continue

        # Find opponent ID
        agent_ids = game_info.get("agent_ids", [])
        opponent_id = next((id for id in agent_ids if id != agent_id), None)

        actions = game_info.get("actions", [])
        for round_actions in actions:
            agent_action = round_actions.get(agent_id)
            opponent_action = round_actions.get(opponent_id)
            if agent_action and opponent_action:
                total_rounds += 1
                if agent_action in ["D", "<Defect>", "<B>"] and opponent_action in [
                    "D",
                    "<Defect>",
                    "<B>",
                ]:
                    mutual_defections += 1

    if total_rounds > 0:
        mutual_defection_rate = (mutual_defections / total_rounds) * 100
    else:
        mutual_defection_rate = 0

    return "mutual_defection_rate", mutual_defection_rate


# def calc_exploitation_rate(data, format_options=None):
#     """
#     Calculates the percentage of rounds where the agent defected while the opponent cooperated.

#     Args:
#         data (list): Raw data files containing game information
#         format_options (list, optional): Formatting options

#     Returns:
#         tuple: ("exploitation_rate", percentage value)
#     """
#     exploitation_rounds = 0
#     total_rounds = 0

#     for game_data in data:
#         game_info = game_data[-1].get("game_info", {})
#         agent_id = game_data[-1].get("agent_id")

#         if not agent_id or not game_info:
#             continue

#         # Find opponent ID
#         agent_ids = game_info.get("agent_ids", [])
#         opponent_id = next((id for id in agent_ids if id != agent_id), None)

#         actions = game_info.get("actions", [])
#         for round_actions in actions:
#             agent_action = round_actions.get(agent_id)
#             opponent_action = round_actions.get(opponent_id)
#             if agent_action and opponent_action:
#                 total_rounds += 1
#                 if agent_action == "D" and opponent_action == "C":
#                     exploitation_rounds += 1

#     if total_rounds > 0:
#         exploitation_rate = (exploitation_rounds / total_rounds) * 100
#     else:
#         exploitation_rate = 0

#     return "exploitation_rate", exploitation_rate


# def calc_sucker_rate(data, format_options=None):
#     """
#     Calculates the percentage of rounds where the agent cooperated while the opponent defected.

#     Args:
#         data (list): Raw data files containing game information
#         format_options (list, optional): Formatting options

#     Returns:
#         tuple: ("sucker_rate", percentage value)
#     """
#     sucker_rounds = 0
#     total_rounds = 0

#     for game_data in data:
#         game_info = game_data[-1].get("game_info", {})
#         agent_id = game_data[-1].get("agent_id")

#         if not agent_id or not game_info:
#             continue

#         # Find opponent ID
#         agent_ids = game_info.get("agent_ids", [])
#         opponent_id = next((id for id in agent_ids if id != agent_id), None)

#         actions = game_info.get("actions", [])
#         for round_actions in actions:
#             agent_action = round_actions.get(agent_id)
#             opponent_action = round_actions.get(opponent_id)
#             if agent_action and opponent_action:
#                 total_rounds += 1
#                 if agent_action == "C" and opponent_action == "D":
#                     sucker_rounds += 1

#     if total_rounds > 0:
#         sucker_rate = (sucker_rounds / total_rounds) * 100
#     else:
#         sucker_rate = 0

#     return "sucker_rate", sucker_rate


# def calc_retaliation_rate(data, format_options=None):
#     """
#     Calculates how often the agent defects after the opponent defects.

#     Args:
#         data (list): Raw data files containing game information
#         format_options (list, optional): Formatting options

#     Returns:
#         tuple: ("retaliation_rate", percentage value)
#     """
#     retaliation_count = 0
#     defection_count = 0

#     for game_data in data:
#         game_info = game_data[-1].get("game_info", {})
#         agent_id = game_data[-1].get("agent_id")

#         if not agent_id or not game_info:
#             continue

#         # Find opponent ID
#         agent_ids = game_info.get("agent_ids", [])
#         opponent_id = next((id for id in agent_ids if id != agent_id), None)

#         actions = game_info.get("actions", [])
#         for i in range(1, len(actions)):  # Skip the first round
#             prev_opponent_action = actions[i - 1].get(opponent_id)
#             current_agent_action = actions[i].get(agent_id)

#             if prev_opponent_action == "D" and current_agent_action:
#                 defection_count += 1
#                 if current_agent_action == "D":
#                     retaliation_count += 1

#     if defection_count > 0:
#         retaliation_rate = (retaliation_count / defection_count) * 100
#     else:
#         retaliation_rate = 0

#     return "retaliation_rate", retaliation_rate


def calc_rounds_count(data, format_options=None):
    """
    Counts the total number of rounds across all games.

    Args:
        data (list): Raw data files containing game information
        format_options (list, optional): Formatting options

    Returns:
        tuple: ("rounds_count", count)
    """
    total_rounds = 0

    for game_data in data:
        game_info = game_data[-1].get("game_info", {})
        if game_info:
            actions = game_info.get("actions", [])
            total_rounds += len(actions)

    return "rounds_count", total_rounds


########################################################
# Methods to gather IPD statistics
########################################################


def get_raw_data_files(path, agent_id):
    """
    Get raw data files from a given path for a specific agent.

    Args:
        path (str): The path to search for raw data files.
        agent_id (str): The ID of the agent to filter files by.
    """
    data = []
    for root, dirs, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            match_pattern = re.search(f".*{agent_id}.*raw_data.*\.json$", file_path)
            if match_pattern:
                with open(file_path, "r") as f:
                    data.append(json.load(f))
    return data


def get_ipd_iteration_stats(data, stat_funcs, format_options=None):
    """
    Processes a list of data files with a set of statistic functions and returns
    the computed statistics.

    Args:
        data (list): A list of raw data files (like match_3_gid_0.json)
        stat_funcs (list): A list of functions that compute statistics
        format_options (list, optional): Options for formatting the output

    Returns:
        dict: A dictionary of computed statistics, where keys are stat names
    """
    stats = {}
    for stat_func in stat_funcs:
        stat_name, stat_value = stat_func(data, format_options)
        stats[stat_name] = stat_value
    return stats


def get_ipd_iterations_stats(iterations_path, agent_id, stat_funcs):
    """
    Get statistics for all iterations of a given agent.

    Args:
        iterations_path (str): The path to the directory containing the iterations
    Returns:
        list: A list of dictionaries of computed statistics, where keys are stat names
    """
    iterations_data = []
    more_iterations = True
    n = 0
    iteration_path = os.path.join(iterations_path, f"iteration_{n:03d}")
    while more_iterations:
        if os.path.isdir(iteration_path):
            data = get_raw_data_files(iteration_path, agent_id)
            iterations_data.append(get_ipd_iteration_stats(data, stat_funcs))
        else:
            more_iterations = False
        n += 1
        iteration_path = os.path.join(iterations_path, f"iteration_{n:03d}")
    return iterations_data


def get_ipd_iterations_stats_tree(iterations_path, agent_id, stat_funcs):
    """
    Get statistics for all iterations of a given agent and organize them in a tree structure.

    Args:
        iterations_path (str): The path to the directory containing the iterations
        agent_id (str): The ID of the agent to get statistics for
        stat_funcs (list): The statistic functions to use

    Returns:
        dict: A tree of statistics for all iterations
    """
    iterations_data = get_ipd_iterations_stats(iterations_path, agent_id, stat_funcs)
    leafstats = iterations_data[0]
    for iteration_data in iterations_data[1:]:
        append_leafstats(leafstats, iteration_data)
    return leafstats


def get_and_save_iterations_stats(
    iterations_path,
    agent_id,
    stat_funcs,
    save=True,
    plot=False,
    plot_EMA=True,
    plot_SMA=True,
    output_path=None,
):
    """
    Get, save and optionally plot statistics for all iterations of a given agent.

    Args:
        iterations_path (str): The path to the directory containing the iterations
        agent_id (str): The ID of the agent to get statistics for
        stat_funcs (list): The statistic functions to use
        save (bool): Whether to save the statistics to a file
        plot (bool): Whether to plot the statistics
        output_path (str): The path to save the statistics to. If None, a default path is used.
    """
    if output_path is None:
        output_path = os.path.join(iterations_path, "0_statistics")
        os.makedirs(output_path, exist_ok=True)
    leafstats = get_ipd_iterations_stats_tree(iterations_path, agent_id, stat_funcs)
    if plot:
        plot_leafstats(leafstats, output_path)
    if plot_EMA:
        plot_EMA_leafstats(leafstats, output_path)
    if plot_SMA:
        plot_SMA_leafstats(leafstats, output_path)
    if save:
        save_leafstats(leafstats, output_path)


if __name__ == "__main__":
    # Get path from command line
    parser = argparse.ArgumentParser(description="Calculate IPD game statistics.")
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the directory containing iterations",
    )
    parser.add_argument(
        "--agent_id", type=str, required=True, help="ID of the agent to analyze"
    )
    parser.add_argument("--plot", action="store_true", help="Plot statistics")
    args = parser.parse_args()

    path = args.path

    stat_functions = [
        calc_cooperation_rate,
        calc_mutual_cooperation_rate,
        calc_mutual_defection_rate,
        get_number_of_rounds,
    ]

    # Run with arguments from command line
    get_and_save_iterations_stats(
        path, args.agent_id, stat_functions, plot=True, plot_EMA=True
    )
    print(f"Done logging IPD statistics for {args.path}.")
