from utils.common_imports import *

def gather_dond_statistics(agent_info, info, stats_to_log):
    """
    Gathers specified statistics of a game for a single agent and outputs them in JSONL format.
    
    Points-based statistics computations have been refactored into external helper functions.
    Additionally, an aggregated statistic is added that sums, over rounds with an agreement,
    the cooperative policy points for both agents and computes their difference.
    
    A new pair of statistics has been added: one for each agent that computes the difference
    between the actual points and the hypothetical cooperative points over all rounds 
    (i.e. irrespective of whether an agreement was reached).
    
    Args:
        agent_info (dict): A dictionary containing agent information.
        info (dict): A dictionary containing game information.
        stats_to_log (list): A list of statistics names to log.
    
    scores:
        dict: A dictionary (formatted like JSONL) containing the specified game statistics.
    """
    statistics = {}
    agent_name = agent_info['agent_name']
    total_points_agent = 0
    total_points_coagent = 0
    total_coop_agent = 0
    total_coop_coagent = 0

    # New accumulators that include all rounds regardless of agreement.
    total_points_agent_all = 0
    total_points_coagent_all = 0
    total_coop_agent_all = 0
    total_coop_coagent_all = 0

    # for each round
    for i, state in enumerate(info['round_agent_roles']):
        agent_role = state.get(agent_name)
        if agent_role is None:
            continue

        coagent_role = next(role for role in state.values() if role != agent_role)
        round_info = {}

        # Extract the agent's own values, the co-agent's values, and the round quantities.
        values = info['round_values'][i][agent_role]
        coagent_values = info['round_values'][i][coagent_role]
        quantities = info['round_quantities'][i]
        points = info['round_points'][i]
        # Note: points for each role are stored in the points dict.
        agent_points = points[agent_role]
        coagent_points = points[coagent_role]
        agreement_reached = info['round_agreements_reached'][i]

        # --- NEW: Compute cooperative points (unconditionally) ---
        coop_agent_all, coop_coagent_all = compute_cooperative_points_for_round(values, coagent_values, quantities)
        total_coop_agent_all += coop_agent_all
        total_coop_coagent_all += coop_coagent_all
        total_points_agent_all += agent_points
        total_points_coagent_all += coagent_points

        # Aggregate cooperative points only if an agreement was reached.
        if agreement_reached:
            coop_agent, coop_coagent = compute_cooperative_points_for_round(values, coagent_values, quantities)
            total_coop_agent += coop_agent
            total_coop_coagent += coop_coagent
            total_points_agent += agent_points
            total_points_coagent += coagent_points

        if "agreement_percentage" in stats_to_log:
            round_info["agreement_percentage"] = 100 if agreement_reached else 0

        if "points" in stats_to_log:
            round_info["points"] = agent_points

        if "coagent_points" in stats_to_log:
            round_info["coagent_points"] = coagent_points

        if "points_difference_on_agreement" in stats_to_log:
            if agreement_reached:
                round_info["points_difference_on_agreement"] = compute_points_difference(agent_points, coagent_points)
            else:
                round_info["points_difference_on_agreement"] = None

        if "imbalance_on_agreement" in stats_to_log:
            if agreement_reached:
                round_info["imbalance_on_agreement"] = calculate_imbalance(points, agent_role, coagent_role)
            else:
                round_info["imbalance_on_agreement"] = None

        if "items_given_to_self" in stats_to_log:
            round_info["items_given_to_self"] = calculate_items_given_to_self(info['round_finalizations'][i][agent_role])

        if "points_on_agreement" in stats_to_log:
            round_info["points_on_agreement"] = compute_points_on_agreement(agent_points, agreement_reached)

        if "coagent_points_on_agreement" in stats_to_log:
            round_info["coagent_points_on_agreement"] = compute_points_on_agreement(coagent_points, agreement_reached)

        if "points_diff_on_agreement" in stats_to_log:
            round_info["points_diff_on_agreement"] = compute_points_diff_on_agreement(agent_points, coagent_points, agreement_reached)

        if "quantities" in stats_to_log:
            round_info["quantities"] = quantities

        if "values" in stats_to_log:
            round_info["values"] = values
        
        if "cooperative_points_difference_on_agreement" in stats_to_log:
            if agreement_reached:
                coop_agent, _ = compute_cooperative_points_for_round(values, coagent_values, quantities)
                round_info["cooperative_points_difference_on_agreement"] = agent_points - coop_agent
            else:
                round_info["cooperative_points_difference_on_agreement"] = None

        if "coagent_cooperative_points_difference_on_agreement" in stats_to_log:
            if agreement_reached:
                _, coop_coagent = compute_cooperative_points_for_round(values, coagent_values, quantities)
                round_info["coagent_cooperative_points_difference_on_agreement"] = coagent_points - coop_coagent
            else:
                round_info["coagent_cooperative_points_difference_on_agreement"] = None

        if "greedy_dominant_points_difference_on_agreement" in stats_to_log:
            if agreement_reached:
                gd_points = compute_greedy_dominant_points(values, quantities)
                round_info["greedy_dominant_points_difference_on_agreement"] = agent_points - gd_points
            else:
                round_info["greedy_dominant_points_difference_on_agreement"] = None

        if "greedy_submission_points_difference_on_agreement" in stats_to_log:
            if agreement_reached:
                gs_points = compute_greedy_submission_points(values)
                round_info["greedy_submission_points_difference_on_agreement"] = agent_points - gs_points
            else:
                round_info["greedy_submission_points_difference_on_agreement"] = None

        if "split_equal_points_difference_on_agreement" in stats_to_log:
            if agreement_reached:
                se_points = compute_split_equal_points(values, quantities)
                round_info["split_equal_points_difference_on_agreement"] = agent_points - se_points
            else:
                round_info["split_equal_points_difference_on_agreement"] = None

        statistics[f"round_{i}"] = round_info

    if "total_coop_points_difference_on_agreement_agent" in stats_to_log:
        if total_points_agent > 0:
            statistics["total_coop_points_difference_on_agreement_agent"] = total_points_agent - total_coop_agent
        else:
            statistics["total_coop_points_difference_on_agreement_agent"] = None

    if "total_coop_points_difference_on_agreement_coagent" in stats_to_log:
        if total_points_coagent > 0:
            statistics["total_coop_points_difference_on_agreement_coagent"] = total_points_coagent - total_coop_coagent
        else:
            statistics["total_coop_points_difference_on_agreement_coagent"] = None

    if "total_imbalance_on_agreement" in stats_to_log:
        if total_points_agent + total_points_coagent > 0:
            statistics["total_imbalance_on_agreement"] = float(abs(total_points_agent - total_points_coagent) / (total_points_agent + total_points_coagent + 1e-6))
        else:
            statistics["total_imbalance_on_agreement"] = None

    if "total_points_difference_on_agreement" in stats_to_log:
        if total_points_agent + total_points_coagent > 0:
            statistics["total_points_difference_on_agreement"] = float(total_points_agent - total_points_coagent)
        else:
            statistics["total_points_difference_on_agreement"] = None
    
    if "total_coop_points_difference_agent" in stats_to_log:
        statistics["total_coop_points_difference_agent"] = total_points_agent_all - total_coop_agent_all

    if "total_coop_points_difference_coagent" in stats_to_log:
        statistics["total_coop_points_difference_coagent"] = total_points_coagent_all - total_coop_coagent_all

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


def calculate_items_given_to_self(finalization):
    """
    Calculates the total items (or value thereof) given to self from finalization data.
    
    Args:
        finalization (dict): A dictionary with values representing item amounts.
    
    scores:
        float or None: The sum of the values if valid; coagentwise, None.
    """
    if not finalization or not all(isinstance(x, (int, float)) for x in finalization.values()):
        return None
    return sum(finalization.values())


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


def compute_cooperative_points_for_round(agent_values, coagent_values, quantities):
    """
    Computes the hypothetical cooperative policy points for both agents for a round.
    The rule used here is symmetric:
      - If agent_values[item] > coagent_values[item]:
          the agent receives full credit (value * quantity) and vice versa.
      - If the values are equal, each receives half credit.
    
    Args:
        agent_values (dict): The agent's values for each item.
        coagent_values (dict): The co-agent's values for each item.
        quantities (dict): The quantities for each item.
    
    scores:
        tuple: (agent_cooperative_points, coagent_cooperative_points)
    """
    agent_coop = 0
    coagent_coop = 0
    for item in quantities:
        if agent_values[item] > coagent_values[item]:
            agent_coop += agent_values[item] * quantities[item]
        elif agent_values[item] < coagent_values[item]:
            coagent_coop += coagent_values[item] * quantities[item]
        else:
            agent_coop += 0.5 * agent_values[item] * quantities[item]
            coagent_coop += 0.5 * coagent_values[item] * quantities[item]
    return agent_coop, coagent_coop


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