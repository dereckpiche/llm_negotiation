from utils.common_imports import *

def gather_dond_statistics(agent_info, info, stats_to_log, format_options=None):
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
    # Set default format options if not provided
    if format_options is None:
        format_options = ["flat"]
    
    statistics = {}
    agent_name = agent_info['agent_name']
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
    for i, state in enumerate(info['round_agent_roles']):
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
        values = info['round_values'][i][agent_role]
        coagent_values = info['round_values'][i][coagent_role]
        quantities = info['round_quantities'][i]
        points = info['round_points'][i]
        # Note: points for each role are stored in the points dict.
        agent_points = points[agent_role]
        coagent_points = points[coagent_role]
        agreement_reached = info['round_agreements_reached'][i]

        # --- NEW: Compute optimal points (unconditionally) ---
        optimal_agent_all, optimal_coagent_all = compute_optimal_points_for_round(values, coagent_values, quantities)
        total_optimal_agent_all += optimal_agent_all
        total_optimal_coagent_all += optimal_coagent_all
        total_points_agent_all += agent_points
        total_points_coagent_all += coagent_points

        # Aggregate optimal points only if an agreement was reached.
        if agreement_reached:
            optimal_agent, optimal_coagent = compute_optimal_points_for_round(values, coagent_values, quantities)
            total_optimal_agent += optimal_agent
            total_optimal_coagent += optimal_coagent
            total_points_agent += agent_points
            total_points_coagent += coagent_points

        # Helper function to add stat to round_info based on format options
        def add_stat(stat_name, agent_value, coagent_value=None):
            if "flat" in format_options:
                round_info["flat"][stat_name] = agent_value
                if coagent_value is not None and f"coagent_{stat_name}" in stats_to_log:
                    round_info["flat"][f"coagent_{stat_name}"] = coagent_value
            
            if "by_agent" in format_options:
                round_info["by_agent"][agent_name][stat_name] = agent_value
                if coagent_name and coagent_value is not None:
                    round_info["by_agent"][coagent_name][stat_name] = coagent_value
            
            if "by_role" in format_options:
                round_info["by_role"][agent_role][stat_name] = agent_value
                if coagent_role and coagent_value is not None:
                    round_info["by_role"][coagent_role][stat_name] = coagent_value
            
            if "by_agent_and_role" in format_options:
                round_info["by_agent_and_role"][agent_name][agent_role][stat_name] = agent_value
                if coagent_name and coagent_value is not None:
                    round_info["by_agent_and_role"][coagent_name][coagent_role][stat_name] = coagent_value

        if "agreement_percentage" in stats_to_log:
            add_stat("agreement_percentage", 100 if agreement_reached else 0)

        if "points" in stats_to_log:
            add_stat("points", agent_points, coagent_points)

        if "points_difference_on_agreement" in stats_to_log:
            if agreement_reached:
                add_stat("points_difference_on_agreement", 
                         compute_points_difference(agent_points, coagent_points),
                         compute_points_difference(coagent_points, agent_points))
            else:
                add_stat("points_difference_on_agreement", None, None)

        if "imbalance_on_agreement" in stats_to_log:
            if agreement_reached:
                imbalance = calculate_imbalance(points, agent_role, coagent_role)
                add_stat("imbalance_on_agreement", imbalance, imbalance)
            else:
                add_stat("imbalance_on_agreement", None, None)

        if "items_given_to_self_percentage" in stats_to_log:
            agent_items = calculate_items_given_to_self(info['round_finalizations'][i].get(agent_role, {}), quantities)
            coagent_items = None
            if coagent_name and coagent_role in info['round_finalizations'][i]:
                coagent_items = calculate_items_given_to_self(info['round_finalizations'][i].get(coagent_role, {}), quantities)
            add_stat("items_given_to_self_percentage", agent_items, coagent_items)

        if "points_on_agreement" in stats_to_log:
            add_stat("points_on_agreement", 
                    compute_points_on_agreement(agent_points, agreement_reached),
                    compute_points_on_agreement(coagent_points, agreement_reached))

        if "points_diff_on_agreement" in stats_to_log:
            add_stat("points_diff_on_agreement", 
                    compute_points_diff_on_agreement(agent_points, coagent_points, agreement_reached),
                    compute_points_diff_on_agreement(coagent_points, agent_points, agreement_reached))

        if "quantities" in stats_to_log:
            add_stat("quantities", quantities, quantities)

        if "values" in stats_to_log:
            add_stat("values", values, coagent_values)
        
        if "optimal_points_difference_on_agreement" in stats_to_log:
            if agreement_reached:
                optimal_agent, optimal_coagent = compute_optimal_points_for_round(values, coagent_values, quantities)
                add_stat("optimal_points_difference_on_agreement", 
                        agent_points - optimal_agent,
                        coagent_points - optimal_coagent)
            else:
                add_stat("optimal_points_difference_on_agreement", None, None)

        if "greedy_dominant_points_difference_on_agreement" in stats_to_log:
            if agreement_reached:
                gd_points = compute_greedy_dominant_points(values, quantities)
                gd_points_coagent = compute_greedy_dominant_points(coagent_values, quantities)
                add_stat("greedy_dominant_points_difference_on_agreement", 
                        agent_points - gd_points,
                        coagent_points - gd_points_coagent)
            else:
                add_stat("greedy_dominant_points_difference_on_agreement", None, None)

        if "greedy_submission_points_difference_on_agreement" in stats_to_log:
            if agreement_reached:
                gs_points = compute_greedy_submission_points(values)
                gs_points_coagent = compute_greedy_submission_points(coagent_values)
                add_stat("greedy_submission_points_difference_on_agreement", 
                        agent_points - gs_points,
                        coagent_points - gs_points_coagent)
            else:
                add_stat("greedy_submission_points_difference_on_agreement", None, None)

        if "split_equal_points_difference_on_agreement" in stats_to_log:
            if agreement_reached:
                se_points = compute_split_equal_points(values, quantities)
                se_points_coagent = compute_split_equal_points(coagent_values, quantities)
                add_stat("split_equal_points_difference_on_agreement", 
                        agent_points - se_points,
                        coagent_points - se_points_coagent)
            else:
                add_stat("split_equal_points_difference_on_agreement", None, None)

        if "more_items_to_value_more_percentage" in stats_to_log:
            if agreement_reached:
                agent_finalization = info['round_finalizations'][i].get(agent_role, {})
                coagent_finalization = info['round_finalizations'][i].get(coagent_role, {})
                is_majority_to_higher_value = check_items_allocation_to_higher_value_agent(
                    values, coagent_values, agent_finalization, coagent_finalization
                )
                add_stat("more_items_to_value_more_percentage", 
                         100 if is_majority_to_higher_value else 0,
                         100 if is_majority_to_higher_value else 0)
            else:
                add_stat("more_items_to_value_more_percentage", None, None)

        statistics[f"round_{i}"] = round_info

    # Initialize totals for each format option
    total_stats = {}
    
    if "flat" in format_options:
        total_stats["flat"] = {}
    
    if "by_agent" in format_options:
        total_stats["by_agent"] = {agent_name: {}}
        if coagent_name:
            total_stats["by_agent"][coagent_name] = {}
    
    if "by_role" in format_options:
        # For totals, we don't have specific roles anymore, so we'll use "overall"
        total_stats["by_role"] = {"agent_overall": {}, "coagent_overall": {}}
    
    if "by_agent_and_role" in format_options:
        total_stats["by_agent_and_role"] = {agent_name: {"overall": {}}}
        if coagent_name:
            total_stats["by_agent_and_role"][coagent_name] = {"overall": {}}

    # Helper function to add total stat based on format options
    def add_total_stat(stat_name, agent_value, coagent_value=None):
        if "flat" in format_options:
            total_stats["flat"][stat_name] = agent_value
            if coagent_value is not None:
                total_stats["flat"][f"coagent_{stat_name}"] = coagent_value
        
        if "by_agent" in format_options:
            total_stats["by_agent"][agent_name][stat_name] = agent_value
            if coagent_name and coagent_value is not None:
                total_stats["by_agent"][coagent_name][stat_name] = coagent_value
        
        if "by_role" in format_options:
            total_stats["by_role"]["agent_overall"][stat_name] = agent_value
            if coagent_value is not None:
                total_stats["by_role"]["coagent_overall"][stat_name] = coagent_value
        
        if "by_agent_and_role" in format_options:
            total_stats["by_agent_and_role"][agent_name]["overall"][stat_name] = agent_value
            if coagent_name and coagent_value is not None:
                total_stats["by_agent_and_role"][coagent_name]["overall"][stat_name] = coagent_value

    if "total_optimal_points_difference_on_agreement_agent" in stats_to_log:
        agent_value = total_points_agent - total_optimal_agent if total_points_agent > 0 else None
        add_total_stat("total_optimal_points_difference_on_agreement", agent_value)

    if "total_optimal_points_difference_on_agreement_coagent" in stats_to_log:
        coagent_value = total_points_coagent - total_optimal_coagent if total_points_coagent > 0 else None
        add_total_stat("total_optimal_points_difference_on_agreement_coagent", coagent_value)

    if "total_imbalance_on_agreement" in stats_to_log:
        if total_points_agent + total_points_coagent > 0:
            imbalance = float(abs(total_points_agent - total_points_coagent) / (total_points_agent + total_points_coagent + 1e-6))
            add_total_stat("total_imbalance_on_agreement", imbalance, imbalance)
        else:
            add_total_stat("total_imbalance_on_agreement", None, None)

    if "total_points_difference_on_agreement" in stats_to_log:
        if total_points_agent + total_points_coagent > 0:
            add_total_stat("total_points_difference_on_agreement", 
                        float(total_points_agent - total_points_coagent),
                        float(total_points_coagent - total_points_agent))
        else:
            add_total_stat("total_points_difference_on_agreement", None, None)
    
    if "total_optimal_points_difference" in stats_to_log:
        add_total_stat("total_optimal_points_difference", 
                      total_points_agent_all - total_optimal_agent_all,
                      total_points_coagent_all - total_optimal_coagent_all)

    if "total_agreement_percentage" in stats_to_log:
        # Calculate the percentage of rounds where an agreement was reached
        total_rounds = len(info['round_agreements_reached'])
        if total_rounds > 0:
            agreements_count = sum(1 for agreement in info['round_agreements_reached'] if agreement)
            agreement_percentage = (agreements_count / total_rounds) * 100
            add_total_stat("total_agreement_percentage", agreement_percentage, agreement_percentage)
        else:
            add_total_stat("total_agreement_percentage", None, None)

    if "more_items_to_value_more_percentage" in stats_to_log:
        # Calculate the total percentage across all rounds
        rounds_with_agreement = sum(1 for i in range(len(info['round_agreements_reached'])) 
                                  if info['round_agreements_reached'][i])
        
        total_higher_value_rounds = 0
        if rounds_with_agreement > 0:
            for i in range(len(info['round_agreements_reached'])):
                if not info['round_agreements_reached'][i]:
                    continue
                
                agent_role = info['round_agent_roles'][i].get(agent_name)
                coagent_name = next((name for name in info['round_agent_roles'][i].keys() if name != agent_name), None)
                coagent_role = info['round_agent_roles'][i].get(coagent_name)
                
                values = info['round_values'][i][agent_role]
                coagent_values = info['round_values'][i][coagent_role]
                
                agent_finalization = info['round_finalizations'][i].get(agent_role, {})
                coagent_finalization = info['round_finalizations'][i].get(coagent_role, {})
                
                is_majority_to_higher_value = check_items_allocation_to_higher_value_agent(
                    values, coagent_values, agent_finalization, coagent_finalization
                )
                
                if is_majority_to_higher_value:
                    total_higher_value_rounds += 1
            
            percentage = (total_higher_value_rounds / rounds_with_agreement) * 100
            add_total_stat("more_items_to_value_more_percentage", percentage, percentage)
        else:
            add_total_stat("more_items_to_value_more_percentage", None, None)

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


def calculate_items_given_to_self(finalization, quantities):
    """
    Calculates the percentage of total available items given to self from finalization data.
    
    Args:
        finalization (dict): A dictionary with values representing item amounts.
        quantities (dict): A dictionary with total quantities of each item.
    
    Returns:
        float or None: The percentage (0-100) of total items given to self if valid; otherwise, None.
    """
    if not finalization or not all(isinstance(x, (int, float)) for x in finalization.values()):
        return None
    
    total_items_allocated = sum(finalization.values())
    total_items_available = sum(quantities.values())
    
    if total_items_available == 0:
        return 0
    
    # Return as a percentage (0-100)
    return 100.0 * total_items_allocated / total_items_available


# --- External points computation functions --- #

def compute_points_difference(agent_points, coagent_points):
    """
    Computes the difference between the agent's points and the coagent agent's points.
    
    Args:
        agent_points (numeric): Points for the agent.
        coagent_points (numeric): Points for the coagent agent.
    
    scores:
        numeric: The difference (agent_points - coagent_points).
    """
    return agent_points - coagent_points


def compute_points_on_agreement(agent_points, agreement_reached):
    """
    scores the agent's points if an agreement was reached, coagentwise None.
    
    Args:
        agent_points (numeric): The agent's points.
        agreement_reached (bool): Whether an agreement was reached.
    
    scores:
        numeric or None
    """
    return agent_points if agreement_reached else None


def compute_points_diff_on_agreement(agent_points, coagent_points, agreement_reached):
    """
    Computes the difference between the agent's and the coagent agent's points
    only if an agreement was reached.
    
    Args:
        agent_points (numeric): The agent's points.
        coagent_points (numeric): The coagent agent's points.
        agreement_reached (bool): Whether an agreement was reached.
    
    scores:
        numeric or None: The difference if agreement reached; coagentwise, None.
    """
    return agent_points - coagent_points if agreement_reached else None


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


def compute_greedy_dominant_points(agent_values, quantities):
    """
    Computes the hypothetical points under a greedy-dominant policy for the agent.
    Policy: Summing the (value * quantity) for all items then subtracting the minimum value.
    
    Args:
        agent_values (dict): The agent's values for each item.
        quantities (dict): The quantities for each item.
    
    scores:
        numeric: The computed greedy dominant points.
    """
    total = sum(agent_values[item] * quantities[item] for item in quantities)
    return total - min(agent_values.values()) if agent_values else total


def compute_greedy_submission_points(agent_values):
    """
    Computes the hypothetical points under a greedy-submission policy by taking the minimum value.
    
    Args:
        agent_values (dict): The agent's values for each item.
    
    scores:
        numeric: The minimum value among the agent's item values.
    """
    return min(agent_values.values()) if agent_values else None


def compute_split_equal_points(agent_values, quantities):
    """
    Computes the hypothetical points under a split-equal policy.
    Policy: Each item contributes half of its full (value * quantity).
    
    Args:
        agent_values (dict): The agent's values for each item.
        quantities (dict): The quantities for each item.
    
    scores:
        numeric: The computed split-equal points.
    """
    return sum(0.5 * agent_values[item] * quantities[item] for item in quantities)


def check_items_allocation_to_higher_value_agent(agent_values, coagent_values, finalization, coagent_finalization):
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
        if (item not in coagent_finalization or 
            item not in agent_values or 
            item not in coagent_values):
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