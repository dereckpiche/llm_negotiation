from utils.common_imports import *
from src.utils.statrees import *




def gather_dond_game_statistics(game_info, stats_to_log, format_options=None):
    """
    Gathers specified statistics of a game for a single agent and outputs them in JSONL format.

    Points-based statistics computations have been refactored into external helper functions.
    Additionally, an aggregated statistic is added that sums, over rounds with an agreement,
    the optimal policy points for both agents and computes their difference.

    A new pair of statistics has been added: one for each agent that computes the difference
    between the actual points and the hypothetical optimal points over all rounds
    (i.e. irrespective of whether an agreement was reached).

    Args:
        agent_info (dict): A dictionary containing agent information.
        info (dict): A dictionary containing game information.
        stats_to_log (list): A list of statistics names to log.
        format_options (list): A list of formatting options to apply. Options:
            - "flat": No nesting (original format)
            - "by_agent": Nest statistics by agent name
            - "by_role": Nest statistics by role
            - "by_agent_and_role": Nest statistics by both agent name and role
            If None, defaults to ["flat"]

    scores:
        dict: A dictionary (formatted like JSONL) containing the specified game statistics.
    """
    info = game_info

    # Set default format options if not provided
    if format_options is None:
        format_options = ["flat"]

    # Define a special flag to indicate a stat is not to be logged (different from None)
    __NO_STAT__ = "__NO_STAT__"

    statistics = {}
    agent_name = agent_info["agent_name"]
    total_points_agent = 0
    total_points_coagent = 0
    total_optimal_agent = 0
    total_optimal_coagent = 0

    # New accumulators that include all rounds regardless of agreement.
    total_points_agent_all = 0
    total_points_coagent_all = 0
    total_optimal_agent_all = 0
    total_optimal_coagent_all = 0

    # for each round
    for i, state in enumerate(info["round_agent_roles"]):
        agent_role = state.get(agent_name)
        if agent_role is None:
            continue

        # Find the co-agent name and role
        coagent_name = next((name for name in state.keys() if name != agent_name), None)
        coagent_role = state.get(coagent_name)

        # Initialize round_info for each format option
        round_info = {}

        # Create structured round info based on format options
        if "flat" in format_options:
            round_info["flat"] = {}

        if "by_agent" in format_options:
            round_info["by_agent"] = {agent_name: {}}
            if coagent_name:
                round_info["by_agent"][coagent_name] = {}

        if "by_role" in format_options:
            round_info["by_role"] = {agent_role: {}}
            if coagent_role:
                round_info["by_role"][coagent_role] = {}

        if "by_agent_and_role" in format_options:
            round_info["by_agent_and_role"] = {agent_name: {agent_role: {}}}
            if coagent_name:
                round_info["by_agent_and_role"][coagent_name] = {coagent_role: {}}

        # Extract the agent's own values, the co-agent's values, and the round quantities.
        values = info["round_values"][i][agent_role]
        coagent_values = info["round_values"][i][coagent_role]
        quantities = info["round_quantities"][i]
        points = info["round_points"][i]
        # Note: points for each role are stored in the points dict.
        agent_points = points[agent_role]
        coagent_points = points[coagent_role]
        agreement_reached = info["round_agreements_reached"][i]

        # --- NEW: Compute optimal points (unconditionally) ---
        optimal_agent_all, optimal_coagent_all = compute_optimal_points_for_round(
            values, coagent_values, quantities
        )
        total_optimal_agent_all += optimal_agent_all
        total_optimal_coagent_all += optimal_coagent_all
        total_points_agent_all += agent_points
        total_points_coagent_all += coagent_points

        # Aggregate optimal points only if an agreement was reached.
        if agreement_reached:
            optimal_agent, optimal_coagent = compute_optimal_points_for_round(
                values, coagent_values, quantities
            )
            total_optimal_agent += optimal_agent
            total_optimal_coagent += optimal_coagent
            total_points_agent += agent_points
            total_points_coagent += coagent_points

        # Helper function to add stat to round_info based on format options
        def add_stat(stat_name, agent_value, coagent_value=None, both_agents=None):
            # Skip values that are marked with the special flag
            if agent_value == __NO_STAT__:
                agent_value = None
            if coagent_value == __NO_STAT__:
                coagent_value = None
            if both_agents == __NO_STAT__:
                both_agents = None

            # Skip if all values are None
            if agent_value is None and coagent_value is None and both_agents is None:
                return

            if "flat" in format_options:
                if both_agents != __NO_STAT__ and both_agents is not None:
                    round_info["flat"][stat_name] = both_agents
                else:
                    if agent_value != __NO_STAT__ and agent_value is not None:
                        round_info["flat"][stat_name] = agent_value
                    if (
                        coagent_value != __NO_STAT__
                        and coagent_value is not None
                        and f"coagent_{stat_name}" in stats_to_log
                    ):
                        round_info["flat"][f"coagent_{stat_name}"] = coagent_value

            if "by_agent" in format_options:
                if both_agents != __NO_STAT__ and both_agents is not None:
                    round_info["by_agent"][stat_name] = both_agents
                else:
                    if agent_value != __NO_STAT__ and agent_value is not None:
                        round_info["by_agent"][agent_name][stat_name] = agent_value
                    if (
                        coagent_name
                        and coagent_value != __NO_STAT__
                        and coagent_value is not None
                    ):
                        round_info["by_agent"][coagent_name][stat_name] = coagent_value

            if "by_role" in format_options:
                if both_agents != __NO_STAT__ and both_agents is not None:
                    round_info["by_role"]["overall"][stat_name] = both_agents
                else:
                    if agent_value != __NO_STAT__ and agent_value is not None:
                        round_info["by_role"][agent_role][stat_name] = agent_value
                    if (
                        coagent_role
                        and coagent_value != __NO_STAT__
                        and coagent_value is not None
                    ):
                        round_info["by_role"][coagent_role][stat_name] = coagent_value

            if "by_agent_and_role" in format_options:
                if both_agents != __NO_STAT__ and both_agents is not None:
                    round_info["by_agent_and_role"][stat_name] = both_agents
                else:
                    if agent_value != __NO_STAT__ and agent_value is not None:
                        round_info["by_agent_and_role"][agent_name][agent_role][
                            stat_name
                        ] = agent_value
                    if (
                        coagent_name
                        and coagent_value != __NO_STAT__
                        and coagent_value is not None
                    ):
                        round_info["by_agent_and_role"][coagent_name][coagent_role][
                            stat_name
                        ] = coagent_value

        # Define functions for each statistic
        def agreement_percentage():
            return (
                100 if agreement_reached else 0,
                100 if agreement_reached else 0,
                __NO_STAT__,
            )

        def round_points():
            return agent_points, coagent_points, __NO_STAT__

        def points_difference_on_agreement():
            if agreement_reached:
                points_difference = agent_points - coagent_points
                return points_difference, -points_difference, __NO_STAT__
            return None, None, __NO_STAT__

        def imbalance_on_agreement():
            if agreement_reached:
                imbalance = calculate_imbalance(points, agent_role, coagent_role)
                return imbalance, imbalance, __NO_STAT__
            return None, None, __NO_STAT__

        def items_given_to_self_percentage():
            agent_items = calculate_items_given_to_self_percentage(
                info["round_finalizations"][i].get(agent_role, {}), quantities
            )
            coagent_items = None
            if coagent_name and coagent_role in info["round_finalizations"][i]:
                coagent_items = calculate_items_given_to_self_percentage(
                    info["round_finalizations"][i].get(coagent_role, {}), quantities
                )
            return agent_items, coagent_items, __NO_STAT__

        def points_on_agreement():
            if agreement_reached:
                return agent_points, coagent_points, __NO_STAT__
            return None, None, __NO_STAT__

        def points_diff_on_agreement():
            if agreement_reached:
                points_diff = agent_points - coagent_points
                return points_diff, -points_diff, __NO_STAT__
            return None, None, __NO_STAT__

        def round_quantities():
            return __NO_STAT__, __NO_STAT__, quantities

        def round_values():
            return values, coagent_values, __NO_STAT__

        def optimal_points_difference_on_agreement():
            if agreement_reached:
                optimal_agent, optimal_coagent = compute_optimal_points_for_round(
                    values, coagent_values, quantities
                )
                return (
                    agent_points - optimal_agent,
                    coagent_points - optimal_coagent,
                    __NO_STAT__,
                )
            return None, None, __NO_STAT__

        def optimal_points_difference():
            optimal_agent, optimal_coagent = compute_optimal_points_for_round(
                values, coagent_values, quantities
            )
            return (
                agent_points - optimal_agent,
                coagent_points - optimal_coagent,
                __NO_STAT__,
            )

        def more_items_to_value_more_percentage():
            agent_finalization = info["round_finalizations"][i].get(agent_role, {})
            coagent_finalization = info["round_finalizations"][i].get(coagent_role, {})
            is_majority_to_higher_value = check_items_allocation_to_higher_value_agent(
                values, coagent_values, agent_finalization, coagent_finalization
            )
            return __NO_STAT__, __NO_STAT__, 100 if is_majority_to_higher_value else 0

        # Calculate and add stats
        for stat_name in stats_to_log:
            if stat_name in locals():
                agent_value, coagent_value, both_agents = locals()[stat_name]()
                add_stat(stat_name, agent_value, coagent_value, both_agents)

        statistics[f"round_{i}"] = round_info

    # Initialize totals for each format option
    total_stats = {}

    if "flat" in format_options:
        total_stats["flat"] = {}

    if "by_agent" in format_options:
        total_stats["by_agent"] = {agent_name: {}}
        if coagent_name:
            total_stats["by_agent"][coagent_name] = {}
        total_stats["by_agent"]["both_agents"] = {}

    if "by_role" in format_options:
        # For totals, we don't have specific roles anymore, so we'll use "overall"
        total_stats["by_role"] = {"agent_overall": {}, "coagent_overall": {}}

    if "by_agent_and_role" in format_options:
        total_stats["by_agent_and_role"] = {agent_name: {"overall": {}}}
        if coagent_name:
            total_stats["by_agent_and_role"][coagent_name] = {"overall": {}}

    # Helper function to add total stat based on format options
    def add_total_stat(stat_name, agent_value, coagent_value=None, both_agents=None):
        # Skip values that are marked with the special flag
        if agent_value == __NO_STAT__:
            agent_value = None
        if coagent_value == __NO_STAT__:
            coagent_value = None
        if both_agents == __NO_STAT__:
            both_agents = None

        # Skip if all values are None
        if agent_value is None and coagent_value is None and both_agents is None:
            return

        if "flat" in format_options:
            if both_agents != __NO_STAT__ and both_agents is not None:
                total_stats["flat"][stat_name] = both_agents
            else:
                if agent_value != __NO_STAT__ and agent_value is not None:
                    total_stats["flat"][stat_name] = agent_value
                if coagent_value != __NO_STAT__ and coagent_value is not None:
                    total_stats["flat"][f"coagent_{stat_name}"] = coagent_value

        if "by_agent" in format_options:
            if both_agents != __NO_STAT__ and both_agents is not None:
                for agent in total_stats["by_agent"]:
                    total_stats["by_agent"]["both_agents"][stat_name] = both_agents
            else:
                if agent_value != __NO_STAT__ and agent_value is not None:
                    total_stats["by_agent"][agent_name][stat_name] = agent_value
                if (
                    coagent_name
                    and coagent_value != __NO_STAT__
                    and coagent_value is not None
                ):
                    total_stats["by_agent"][coagent_name][stat_name] = coagent_value

        if "by_role" in format_options:
            if both_agents != __NO_STAT__ and both_agents is not None:
                total_stats["by_role"]["overall"][stat_name] = both_agents
            else:
                if agent_value != __NO_STAT__ and agent_value is not None:
                    total_stats["by_role"]["agent_overall"][stat_name] = agent_value
                if coagent_value != __NO_STAT__ and coagent_value is not None:
                    total_stats["by_role"]["coagent_overall"][stat_name] = coagent_value

        if "by_agent_and_role" in format_options:
            if both_agents != __NO_STAT__ and both_agents is not None:
                for agent in total_stats["by_agent_and_role"]:
                    total_stats["by_agent_and_role"][agent]["overall"][
                        stat_name
                    ] = both_agents
            else:
                if agent_value != __NO_STAT__ and agent_value is not None:
                    total_stats["by_agent_and_role"][agent_name]["overall"][
                        stat_name
                    ] = agent_value
                if (
                    coagent_name
                    and coagent_value != __NO_STAT__
                    and coagent_value is not None
                ):
                    total_stats["by_agent_and_role"][coagent_name]["overall"][
                        stat_name
                    ] = coagent_value

    def total_imbalance_on_agreement():
        imbalance = float(
            abs(total_points_agent - total_points_coagent)
            / (total_points_agent + total_points_coagent + 1e-6)
        )
        return imbalance, imbalance, __NO_STAT__

    def total_points_difference_on_agreement():
        if total_points_agent + total_points_coagent > 0:
            return (
                float(total_points_agent - total_points_coagent),
                float(total_points_coagent - total_points_agent),
                __NO_STAT__,
            )
        return None, None, __NO_STAT__

    def total_optimal_points_difference():
        return (
            total_points_agent_all - total_optimal_agent_all,
            total_points_coagent_all - total_optimal_coagent_all,
            __NO_STAT__,
        )

    def total_agreement_percentage():
        total_rounds = len(info["round_agreements_reached"])
        if total_rounds > 0:
            agreements_count = sum(
                1 for agreement in info["round_agreements_reached"] if agreement
            )
            agreement_percentage = (agreements_count / total_rounds) * 100
            return agreement_percentage, agreement_percentage, __NO_STAT__
        return None, None, __NO_STAT__

    def total_sum_points_percentage_of_max():
        max_points = total_optimal_agent_all + total_optimal_coagent_all
        if max_points > 0:
            sum_points_percentage = (
                100 * (total_points_agent_all + total_points_coagent_all) / max_points
            )
            return __NO_STAT__, __NO_STAT__, sum_points_percentage
        return __NO_STAT__, __NO_STAT__, None

    def total_imbalance():
        total_points = total_points_agent_all + total_points_coagent_all
        if total_points > 0:
            imbalance = (
                abs(total_points_agent_all - total_points_coagent_all) / total_points
            )
            return __NO_STAT__, __NO_STAT__, imbalance
        return __NO_STAT__, __NO_STAT__, None

    def total_average_imbalance():
        """
        Computes the average imbalance across all rounds.
        """
        imbalances = []
        for i, state in enumerate(info["round_agent_roles"]):
            agent_role = state.get(agent_name)
            if agent_role is None:
                continue

            # Find the co-agent name and role
            coagent_name_i = next(
                (name for name in state.keys() if name != agent_name), None
            )
            coagent_role = state.get(coagent_name_i)
            if coagent_role is None:
                continue

            points = info["round_points"][i]
            # Calculate imbalance for this round
            round_imbalance = calculate_imbalance(points, agent_role, coagent_role)
            imbalances.append(round_imbalance)

        avg_imbalance = sum(imbalances) / len(imbalances) if imbalances else None
        # Return tuple of (agent_value, coagent_value, both_agent_value)
        return avg_imbalance, avg_imbalance, __NO_STAT__

    def total_more_items_to_value_more_percentage():
        total_higher_value_rounds = 0
        nb_rounds = len(info["round_agreements_reached"])
        for i in range(nb_rounds):
            if i >= len(info["round_agent_roles"]):
                continue
            agent_role = info["round_agent_roles"][i].get(agent_name)
            if agent_role is None:
                continue
            coagent_name_i = next(
                (
                    name
                    for name in info["round_agent_roles"][i].keys()
                    if name != agent_name
                ),
                None,
            )
            coagent_role_i = info["round_agent_roles"][i].get(coagent_name_i)
            if coagent_role_i is None:
                continue
            values_i = info["round_values"][i][agent_role]
            coagent_values_i = info["round_values"][i][coagent_role_i]
            agent_finalization = info["round_finalizations"][i].get(agent_role, {})
            coagent_finalization = info["round_finalizations"][i].get(
                coagent_role_i, {}
            )
            is_majority_to_higher_value = check_items_allocation_to_higher_value_agent(
                values_i, coagent_values_i, agent_finalization, coagent_finalization
            )
            if is_majority_to_higher_value:
                total_higher_value_rounds += 1
        percentage = (
            (total_higher_value_rounds / nb_rounds) * 100 if nb_rounds > 0 else None
        )
        return __NO_STAT__, __NO_STAT__, percentage

    def total_items_given_to_self_percentage():
        total_items_allocated = 0
        total_items_available = 0
        for i in range(len(info["round_finalizations"])):
            if i >= len(info["round_agent_roles"]):
                continue
            agent_role_i = info["round_agent_roles"][i].get(agent_name)
            if agent_role_i is None:
                continue
            finalization = info["round_finalizations"][i].get(agent_role_i, {})
            quantities_i = info["round_quantities"][i]
            try:
                # Convert all values to integers before summing
                total_items_allocated += sum(int(val) for val in finalization.values())
                total_items_available += sum(quantities_i.values())
            except (ValueError, TypeError):
                # Skip this round if any value can't be converted to int
                continue
        if total_items_available > 0:
            total_percentage = 100.0 * total_items_allocated / total_items_available
            return total_percentage, __NO_STAT__, __NO_STAT__
        return None, __NO_STAT__, __NO_STAT__

    def number_of_rounds():
        return len(info["round_agent_roles"]), __NO_STAT__, __NO_STAT__

    # Calculate and add total stats
    for stat_name in stats_to_log:
        # Handle total stats
        if stat_name in locals():
            agent_value, coagent_value, both_agents = locals()[stat_name]()
            add_total_stat(stat_name, agent_value, coagent_value, both_agents)

    statistics["totals"] = total_stats

    return statistics


def calculate_imbalance(points, agent_role, coagent_role):
    """
    Calculates the imbalance between the points of the agent and the coagent agent.

    Args:
        points (dict): A dictionary containing points for each role.
        agent_role (str): The role of the agent.
        coagent_role (str): The role of the coagent agent.

    scores:
        float: The calculated imbalance.
    """
    total_points = points[agent_role] + points[coagent_role]
    if total_points == 0:
        return 0
    return abs((points[agent_role] - points[coagent_role]) / total_points)


def calculate_items_given_to_self_percentage(finalization, quantities):
    """
    Calculates the percentage of total available items given to self from finalization data.

    Args:
        finalization (dict): A dictionary with values representing item amounts.
        quantities (dict): A dictionary with total quantities of each item.

    Returns:
        float or None: The percentage (0-100) of total items given to self if valid; otherwise, None.
    """
    if not finalization or not all(
        isinstance(x, (int, float)) for x in finalization.values()
    ):
        return None

    total_items_allocated = sum(finalization.values())
    total_items_available = sum(quantities.values())

    if total_items_available == 0:
        return 0

    # Return as a percentage (0-100)
    return 100.0 * total_items_allocated / total_items_available


def compute_optimal_points_for_round(agent_values, coagent_values, quantities):
    """
    Computes the hypothetical optimal policy points for both agents for a round.
    The rule used here is symmetric:
      - If agent_values[item] > coagent_values[item]:
          the agent receives full credit (value * quantity) and vice versa.
      - If the values are equal, each receives half credit.

    Args:
        agent_values (dict): The agent's values for each item.
        coagent_values (dict): The co-agent's values for each item.
        quantities (dict): The quantities for each item.

    scores:
        tuple: (agent_optimal_points, coagent_optimal_points)
    """
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


def check_items_allocation_to_higher_value_agent(
    agent_values, coagent_values, finalization, coagent_finalization
):
    """
    Checks if the majority of items went to the agent that values them more.

    Args:
        agent_values (dict): The agent's values for each item.
        coagent_values (dict): The co-agent's values for each item.
        finalization (dict): The agent's finalization (items taken).
        coagent_finalization (dict): The co-agent's finalization (items taken).

    Returns:
        int: 1 if majority of items went to agents valuing them more, 0 otherwise, None if finalization is invalid.
    """
    if not finalization or not coagent_finalization:
        return None

    total_items = 0
    items_to_higher_value_agent = 0

    for item in finalization:
        # Skip if item not in all dictionaries
        if (
            item not in coagent_finalization
            or item not in agent_values
            or item not in coagent_values
        ):
            continue

        agent_count = finalization.get(item, 0)
        coagent_count = coagent_finalization.get(item, 0)
        total_items += agent_count + coagent_count

        # Check which agent values this item more
        if agent_values[item] > coagent_values[item]:
            # Agent values item more, so add agent's count to correct allocation
            items_to_higher_value_agent += agent_count
        elif coagent_values[item] > agent_values[item]:
            # Co-agent values item more, so add co-agent's count to correct allocation
            items_to_higher_value_agent += coagent_count
        else:
            # Equal values, so add half of each agent's count
            items_to_higher_value_agent += (agent_count + coagent_count) / 2

    # Check if majority of items went to agents valuing them more
    return 1 if total_items > 0 and items_to_higher_value_agent > total_items / 2 else 0


def generate_frequency_counts(input_path):
    agreement_percent_values = []
    items_given_to_self_values = []

    for filename in os.listdir(input_path):
        if filename.endswith(".json"):
            file_path = os.path.join(input_path, filename)
            with open(file_path, "r") as f:
                data = json.load(f)

            for rounds, values in data.items():
                agreement_percent_values.append(values["agreement_percentage"])
                items_given_to_self_values.append(values["items_given_to_self"])

    # Convert lists to frequency counts
    agreement_percent_freq_counts = dict(Counter(agreement_percent_values))
    items_given_to_self_freq_counts = dict(Counter(items_given_to_self_values))

    # Combine into a final dictionary
    freq_stats = {
        "agreement_percent_freq": agreement_percent_freq_counts,
        "items_given_to_self_freq": items_given_to_self_freq_counts,
    }

    output_path = os.path.join(input_path, "frequency_stats.json")
    # Save to JSON
    with open(output_path, "w") as f:
        json.dump(freq_stats, f, indent=4)
