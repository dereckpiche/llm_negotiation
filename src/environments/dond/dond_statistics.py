"""
Deal or No Deal (DOND) Game Statistics

This module provides a modular approach to calculating statistics for Deal or No Deal
game data. The original monolithic approach has been replaced with a set of smaller,
focused statistic calculation functions that can be composed to create custom
statistical analyses.

Key features:
- Modular stat calculation functions that each return (stat_name, stat_value)
- A flexible stat aggregation function (get_dond_iteration_stats)
- Formatting utilities for organizing the stats
- Backwards compatibility through a wrapper function (gather_dond_game_statistics_modular)

Usage:
    # Load data
    data = get_raw_data_files(path, agent_id)

    # Select which stat functions to use
    stat_functions = [
        calc_total_agreement_percentage,
        calc_total_points,
        calc_agreement_imbalance
    ]

    # Calculate statistics
    stats = get_dond_iteration_stats(data, stat_functions)

    # Format if needed
    formatted_stats = format_stats(stats, ["by_agent"])
"""

import re

from utils.common_imports import *
from utils.leafstats import *

############################################################
# Modular Statistics functions
############################################################


def get_proposal_frequency_stats(data, format_options=None):
    """
    Calculates the frequency of proposals in the data by item category.

    Args:
        data (list): Raw data files containing game information
        format_options (list, optional): Formatting options

    Returns:
        tuple: ("proposal_frequency_by_category", {category: count})
    """
    proposal_counts = {}

    for game_data in data:
        game_info = game_data[-1].get("game_info", {})
        agent_name = game_data[-1].get("agent_name")

        if not agent_name or not game_info:
            continue

        # Process each round
        for i, state in enumerate(game_info["round_agent_roles"]):
            agent_role = state.get(agent_name)
            round_role = game_info["round_agent_roles"][i].get(agent_name)
            round_proposal = game_info["round_finalizations"][i]
            agent_proposal = round_proposal.get(round_role, {})

            for item, quantity in agent_proposal.items():
                if item not in proposal_counts:
                    proposal_counts[item] = {}
                if quantity not in proposal_counts[item]:
                    proposal_counts[item][quantity] = 0
                proposal_counts[item][quantity] += 1

    return "proposal_frequency_by_category", proposal_counts


# Modular stat functions - each returns (stat_name, stat_value)


def calc_total_agreement_percentage(data, format_options=None):
    """
    Calculates the percentage of rounds where an agreement was reached.

    Args:
        data (list): Raw data files containing game information
        format_options (list, optional): Formatting options

    Returns:
        tuple: ("total_agreement_percentage", percentage value)
    """
    total_agreements = 0
    total_rounds = 0

    for game_data in data:
        game_info = game_data[-1].get("game_info", {})
        agreements = game_info.get("round_agreements_reached", [])
        total_rounds += len(agreements)
        total_agreements += sum(1 for agreement in agreements if agreement)

    if total_rounds > 0:
        agreement_percentage = (total_agreements / total_rounds) * 100
    else:
        agreement_percentage = 0

    return "total_agreement_percentage", agreement_percentage


def calc_total_points(data, format_options=None):
    """
    Calculates the total points for both the agent and co-agent across all games.

    Args:
        data (list): Raw data files containing game information
        format_options (list, optional): Formatting options

    Returns:
        tuple: ("total_points", {"agent": points, "coagent": points})
    """
    agent_points = 0
    coagent_points = 0

    for game_data in data:
        game_info = game_data[-1].get("game_info", {})
        agent_name = game_data[-1].get("agent_name")

        if not agent_name or not game_info:
            continue

        # Process each round
        for i, state in enumerate(game_info["round_agent_roles"]):
            agent_role = state.get(agent_name)
            if not agent_role:
                continue

            # Find co-agent
            coagent_name = next(
                (name for name in state.keys() if name != agent_name), None
            )
            coagent_role = state.get(coagent_name)

            if i < len(game_info["round_points"]):
                round_points = game_info["round_points"][i]
                agent_points += round_points.get(agent_role, 0)
                coagent_points += round_points.get(coagent_role, 0)

    return "total_points", {"agent": agent_points, "coagent": coagent_points}


def calc_agreement_imbalance(data, format_options=None):
    """
    Calculates the imbalance between agent and co-agent points on agreements.

    Args:
        data (list): Raw data files containing game information
        format_options (list, optional): Formatting options

    Returns:
        tuple: ("agreement_imbalance", imbalance value)
    """
    agent_points = 0
    coagent_points = 0

    for game_data in data:
        game_info = game_data[-1].get("game_info", {})
        agent_name = game_data[-1].get("agent_name")

        if not agent_name or not game_info:
            continue

        # Process each round
        for i, state in enumerate(game_info["round_agent_roles"]):
            # Skip if not an agreement
            if (
                i >= len(game_info["round_agreements_reached"])
                or not game_info["round_agreements_reached"][i]
            ):
                continue

            agent_role = state.get(agent_name)
            if not agent_role:
                continue

            # Find co-agent
            coagent_name = next(
                (name for name in state.keys() if name != agent_name), None
            )
            coagent_role = state.get(coagent_name)

            if i < len(game_info["round_points"]):
                round_points = game_info["round_points"][i]
                agent_points += round_points.get(agent_role, 0)
                coagent_points += round_points.get(coagent_role, 0)

    total_points = agent_points + coagent_points
    if total_points > 0:
        imbalance = abs(agent_points - coagent_points) / total_points
    else:
        imbalance = 0

    return "agreement_imbalance", imbalance


def calc_items_allocation_efficiency(data, format_options=None):
    """
    Calculates how efficiently items
    were allocated to the agent that values them more.

    Args:
        data (list): Raw data files containing game information
        format_options (list, optional): Formatting options

    Returns:
        tuple: ("items_allocation_efficiency", percentage value)
    """

    # Sub-method that checks if majority of items got attributed
    # to the agent that values them more
    def check_items_allocation_to_higher_value_agent(
        agent_values, coagent_values, finalization, coagent_finalization
    ):
        if not finalization or not coagent_finalization:
            return None
        total_items = 0
        items_to_higher_value_agent = 0
        for item in finalization:
            if (
                item not in coagent_finalization
                or item not in agent_values
                or item not in coagent_values
            ):
                continue
            agent_count = finalization.get(item, 0)
            coagent_count = coagent_finalization.get(item, 0)
            total_items += agent_count + coagent_count
            if agent_values[item] > coagent_values[item]:
                items_to_higher_value_agent += agent_count
            elif coagent_values[item] > agent_values[item]:
                items_to_higher_value_agent += coagent_count
            else:
                items_to_higher_value_agent += (agent_count + coagent_count) / 2

        return (
            1
            if total_items > 0 and items_to_higher_value_agent > total_items / 2
            else 0
        )

    total_higher_value_rounds = 0
    total_rounds = 0

    for game_data in data:
        game_info = game_data[-1].get("game_info", {})
        agent_name = game_data[-1].get("agent_name")

        if not agent_name or not game_info:
            continue

        # Process each round with agreements
        for i, state in enumerate(game_info["round_agent_roles"]):
            if (
                i >= len(game_info["round_agreements_reached"])
                or not game_info["round_agreements_reached"][i]
            ):
                continue

            agent_role = state.get(agent_name)
            if not agent_role:
                continue

            # Find co-agent
            coagent_name = next(
                (name for name in state.keys() if name != agent_name), None
            )
            coagent_role = state.get(coagent_name)

            if i < len(game_info["round_values"]) and i < len(
                game_info["round_finalizations"]
            ):
                values = game_info["round_values"][i][agent_role]
                coagent_values = game_info["round_values"][i][coagent_role]

                agent_finalization = game_info["round_finalizations"][i].get(
                    agent_role, {}
                )
                coagent_finalization = game_info["round_finalizations"][i].get(
                    coagent_role, {}
                )

                is_efficient = check_items_allocation_to_higher_value_agent(
                    values, coagent_values, agent_finalization, coagent_finalization
                )

                if is_efficient:
                    total_higher_value_rounds += 1

                total_rounds += 1

    if total_rounds > 0:
        efficiency_percentage = (total_higher_value_rounds / total_rounds) * 100
    else:
        efficiency_percentage = 0

    return "items_allocation_efficiency", efficiency_percentage


def calc_optimal_points_diff(data, format_options=None):
    """
    Calculates the difference between actual points and optimal points.

    Args:
        data (list): Raw data files containing game information
        format_options (list, optional): Formatting options

    Returns:
        tuple: ("optimal_points_diff", {"agent": diff, "coagent": diff})
    """
    agent_actual = 0
    coagent_actual = 0
    agent_optimal = 0
    coagent_optimal = 0

    for game_data in data:
        game_info = game_data[-1].get("game_info", {})
        agent_name = game_data[-1].get("agent_name")

        if not agent_name or not game_info:
            continue

        # Process each round
        for i, state in enumerate(game_info["round_agent_roles"]):
            agent_role = state.get(agent_name)
            if not agent_role:
                continue

            # Find co-agent
            coagent_name = next(
                (name for name in state.keys() if name != agent_name), None
            )
            coagent_role = state.get(coagent_name)

            if (
                i < len(game_info["round_values"])
                and i < len(game_info["round_quantities"])
                and i < len(game_info["round_points"])
            ):
                values = game_info["round_values"][i][agent_role]
                coagent_values = game_info["round_values"][i][coagent_role]
                quantities = game_info["round_quantities"][i]
                round_points = game_info["round_points"][i]

                # Calculate optimal points
                opt_agent, opt_coagent = compute_optimal_points_for_round(
                    values, coagent_values, quantities
                )

                # Accumulate actual points
                agent_actual += round_points.get(agent_role, 0)
                coagent_actual += round_points.get(coagent_role, 0)

                # Accumulate optimal points
                agent_optimal += opt_agent
                coagent_optimal += opt_coagent

    agent_diff = agent_actual - agent_optimal
    coagent_diff = coagent_actual - coagent_optimal

    return "optimal_points_diff", {"agent": agent_diff, "coagent": coagent_diff}


def calc_items_given_to_self(data, format_options=None):
    """
    Calculates the percentage of items an agent gave to itself across all games.

    Args:
        data (list): Raw data files containing game information
        format_options (list, optional): Formatting options

    Returns:
        tuple: ("items_given_to_self_percentage", percentage value)
    """
    total_items_allocated = 0
    total_items_available = 0

    for game_data in data:
        game_info = game_data[-1].get("game_info", {})
        agent_name = game_data[-1].get("agent_name")

        if not agent_name or not game_info:
            continue

        # Process each round
        for i, state in enumerate(game_info["round_agent_roles"]):
            agent_role = state.get(agent_name)
            if not agent_role:
                continue

            if i >= len(game_info["round_finalizations"]) or i >= len(
                game_info["round_quantities"]
            ):
                continue

            finalization = game_info["round_finalizations"][i].get(agent_role, {})
            quantities = game_info["round_quantities"][i]

            try:
                # Convert all values to integers before summing
                total_items_allocated += sum(int(val) for val in finalization.values())
                total_items_available += sum(quantities.values())
            except (ValueError, TypeError):
                # Skip this round if any value can't be converted to int
                continue

    if total_items_available > 0:
        percentage = 100.0 * total_items_allocated / total_items_available
    else:
        percentage = 0

    return "items_given_to_self_percentage", percentage


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
            total_rounds += len(game_info.get("round_agent_roles", []))

    return "rounds_count", total_rounds


def calc_round_stats(data, format_options=None):
    """
    Calculates per-round statistics.

    Args:
        data (list): Raw data files containing game information
        format_options (list, optional): Formatting options

    Returns:
        tuple: ("round_stats", [list of per-round stats])
    """
    round_stats = []

    for game_data in data:
        game_info = game_data[-1].get("game_info", {})
        agent_name = game_data[-1].get("agent_name")

        if not agent_name or not game_info:
            continue

        # Process each round
        for i, state in enumerate(game_info["round_agent_roles"]):
            agent_role = state.get(agent_name)
            if not agent_role:
                continue

            # Find co-agent
            coagent_name = next(
                (name for name in state.keys() if name != agent_name), None
            )
            coagent_role = state.get(coagent_name)

            # Initialize round data
            round_data = {
                "round_number": i,
                "agreement_reached": False,
                "agent_points": 0,
                "coagent_points": 0,
                "imbalance": 0,
            }

            # Add agreement information
            if i < len(game_info["round_agreements_reached"]):
                round_data["agreement_reached"] = game_info["round_agreements_reached"][
                    i
                ]

            # Add points information
            if i < len(game_info["round_points"]):
                round_points = game_info["round_points"][i]
                round_data["agent_points"] = round_points.get(agent_role, 0)
                round_data["coagent_points"] = round_points.get(coagent_role, 0)

                # Calculate imbalance
                total_points = round_data["agent_points"] + round_data["coagent_points"]
                if total_points > 0:
                    round_data["imbalance"] = (
                        abs(round_data["agent_points"] - round_data["coagent_points"])
                        / total_points
                    )

            round_stats.append(round_data)

    return "round_stats", round_stats


def calc_sum_points_percentage_of_max(data, format_options=None):
    """
    Calculates the total sum of points (agent + coagent) as a percentage of the maximum possible points.

    Args:
        data (list): Raw data files containing game information
        format_options (list, optional): Formatting options

    Returns:
        tuple: ("sum_points_percentage_of_max", percentage value)
    """

    def compute_optimal_points_for_round(agent_values, coagent_values, quantities):
        agent_optimal = 0
        coagent_optimal = 0
        for item in quantities:
            if agent_values[item] > coagent_values[item]:
                agent_optimal += agent_values[item] * quantities[item]
            elif agent_values[item] < coagent_values[item]:
                coagent_optimal += coagent_values[item] * quantities[item]
            else:
                agent_optimal += 0.5 * agent_values[item] * quantities[item]
                coagent_optimal += 0.5 * coagent_values[item] * quantities[item]
        return agent_optimal, coagent_optimal

    total_points_agent = 0
    total_points_coagent = 0
    total_optimal_agent = 0
    total_optimal_coagent = 0

    for game_data in data:
        game_info = game_data[-1].get("game_info", {})
        agent_name = game_data[-1].get("agent_name")

        if not agent_name or not game_info:
            continue

        # Process each round
        for i, state in enumerate(game_info["round_agent_roles"]):
            agent_role = state.get(agent_name)
            if not agent_role:
                continue

            # Find co-agent
            coagent_name = next(
                (name for name in state.keys() if name != agent_name), None
            )
            coagent_role = state.get(coagent_name)

            if (
                i < len(game_info["round_values"])
                and i < len(game_info["round_quantities"])
                and i < len(game_info["round_points"])
            ):
                values = game_info["round_values"][i][agent_role]
                coagent_values = game_info["round_values"][i][coagent_role]
                quantities = game_info["round_quantities"][i]
                round_points = game_info["round_points"][i]

                # Calculate optimal points
                opt_agent, opt_coagent = compute_optimal_points_for_round(
                    values, coagent_values, quantities
                )

                # Accumulate actual points
                total_points_agent += round_points.get(agent_role, 0)
                total_points_coagent += round_points.get(coagent_role, 0)

                # Accumulate optimal points
                total_optimal_agent += opt_agent
                total_optimal_coagent += opt_coagent

    # Calculate total points and max points
    total_points = total_points_agent + total_points_coagent
    max_points = total_optimal_agent + total_optimal_coagent

    # Calculate percentage
    if max_points > 0:
        percentage = 100 * total_points / max_points
    else:
        percentage = 0

    return "sum_points_percentage_of_max", percentage


########################################################
# Methods to gather DoND statistics
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


def get_dond_iteration_stats(data, stat_funcs, format_options=None):
    """
    Processes a list of data files with a set of statistic functions and returns
    the computed statistics.

    Args:
        data (list): A list of raw data files (like match_1.json)
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


def get_dond_iterations_stats(iterations_path, agent_id, stat_funcs):
    """
    Get statistics for all iterations of a given agent.

    Args:
        iterations_path (str): The path to the directory containing the iterations
    Returns:
        list: A list of dictionaries of computed statistics, where keys are stat names
    """
    iterations_data = []
    more_iterations = True
    paths = os.listdir(iterations_path)
    n = 0
    iteration_path = os.path.join(iterations_path, f"iteration_{n:03d}")
    while more_iterations:
        if os.path.isdir(iteration_path):
            data = get_raw_data_files(iteration_path, agent_id)
            iterations_data.append(get_dond_iteration_stats(data, stat_funcs))
        else:
            more_iterations = False
        n += 1
        iteration_path = os.path.join(iterations_path, f"iteration_{n:03d}")
    return iterations_data


def get_dond_iterations_stats_tree(iterations_path, agent_id, stat_funcs):
    """
    Get statistics for all iterations of a given agent.

    Args:
        iterations_path (str): The path to the directory containing the iterations
    """
    iterations_data = get_dond_iterations_stats(iterations_path, agent_id, stat_funcs)
    leafstats = iterations_data[0]
    for iteration_data in iterations_data[1:]:
        append_leafstats(leafstats, iteration_data)
    return leafstats


def get_and_save_iterations_stats(
    iterations_path, agent_id, stat_funcs, save=True, plot=False, output_path=None
):
    if output_path is None:
        output_path = os.path.join(iterations_path, "0_statistics")
        os.makedirs(output_path, exist_ok=True)
    leafstats = get_dond_iterations_stats_tree(iterations_path, agent_id, stat_funcs)
    if plot:
        plot_leafstats(leafstats, output_path)
    if save:
        save_leafstats(leafstats, output_path)


if __name__ == "__main__":
    path = "/home/mila/d/dereck.piche/scratch/llm_negotiation/REPRODUCE/greedy/seed_645"

    stat_functions = [
        calc_sum_points_percentage_of_max,
        calc_items_given_to_self,
        get_proposal_frequency_stats,
    ]

    # Calculate statistics
    get_and_save_iterations_stats(
        iterations_path=path, agent_id="Alice", stat_funcs=stat_functions, plot=True
    )

    print("Done")
