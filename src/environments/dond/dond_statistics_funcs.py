from utils.common_imports import *

def gather_dond_statistics(player_info, info, stats_to_log):
    """
    Gathers specified statistics of a game for a single player and outputs them in JSONL format.
    
    Points-based statistics computations have been refactored into external helper functions.
    Additionally, an aggregated statistic is added that sums, over rounds with an agreement,
    the cooperative policy points for both agents and computes their difference.
    
    A new pair of statistics has been added: one for each player that computes the difference
    between the actual points and the hypothetical cooperative points over all rounds 
    (i.e. irrespective of whether an agreement was reached).
    
    Args:
        player_info (dict): A dictionary containing player information.
        info (dict): A dictionary containing game information.
        stats_to_log (list): A list of statistics names to log.
    
    scores:
        dict: A dictionary (formatted like JSONL) containing the specified game statistics.
    """
    statistics = {}
    player_name = player_info['player_name']
    total_points_player = 0
    total_points_coplayer = 0
    total_coop_player = 0
    total_coop_coplayer = 0

    # New accumulators that include all rounds regardless of agreement.
    total_points_player_all = 0
    total_points_coplayer_all = 0
    total_coop_player_all = 0
    total_coop_coplayer_all = 0

    # for each round
    for i, state in enumerate(info['round_player_roles']):
        player_role = state.get(player_name)
        if player_role is None:
            continue

        other_role = next(role for role in state.values() if role != player_role)
        round_info = {}
        
        # Extract the player's own values, the co-player's values, and the round quantities.
        values = info['round_values'][i][player_role]
        coplayer_values = info['round_values'][i][other_role]
        quantities = info['round_quantities'][i]
        points = info['round_points'][i]
        # Note: points for each role are stored in the points dict.
        player_points = points[player_role]
        other_points = points[other_role]
        agreement_reached = info['round_agreements_reached'][i]

        # --- NEW: Compute cooperative points (unconditionally) ---
        coop_player_all, coop_coplayer_all = compute_cooperative_points_for_round(values, coplayer_values, quantities)
        total_coop_player_all += coop_player_all
        total_coop_coplayer_all += coop_coplayer_all
        total_points_player_all += player_points
        total_points_coplayer_all += other_points

        # Aggregate cooperative points only if an agreement was reached.
        if agreement_reached:
            coop_player, coop_coplayer = compute_cooperative_points_for_round(values, coplayer_values, quantities)
            total_coop_player += coop_player
            total_coop_coplayer += coop_coplayer
            total_points_player += player_points
            total_points_coplayer += other_points

        if "agreement_percentage" in stats_to_log:
            round_info["agreement_percentage"] = 100 if agreement_reached else 0

        if "points" in stats_to_log:
            round_info["points"] = player_points

        if "other_points" in stats_to_log:
            round_info["other_points"] = other_points

        if "points_difference_on_agreement" in stats_to_log:
            if agreement_reached:
                round_info["points_difference_on_agreement"] = compute_points_difference(player_points, other_points)
            else:
                round_info["points_difference_on_agreement"] = None

        if "imbalance_on_agreement" in stats_to_log:
            if agreement_reached:
                round_info["imbalance_on_agreement"] = calculate_imbalance(points, player_role, other_role)
            else:
                round_info["imbalance_on_agreement"] = None

        if "items_given_to_self" in stats_to_log:
            round_info["items_given_to_self"] = calculate_items_given_to_self(info['round_finalizations'][i][player_role])

        if "points_on_agreement" in stats_to_log:
            round_info["points_on_agreement"] = compute_points_on_agreement(player_points, agreement_reached)

        if "other_points_on_agreement" in stats_to_log:
            round_info["other_points_on_agreement"] = compute_points_on_agreement(other_points, agreement_reached)

        if "points_diff_on_agreement" in stats_to_log:
            round_info["points_diff_on_agreement"] = compute_points_diff_on_agreement(player_points, other_points, agreement_reached)

        if "quantities" in stats_to_log:
            round_info["quantities"] = quantities

        if "values" in stats_to_log:
            round_info["values"] = values
        
        if "points_difference_on_agreement" in stats_to_log:
            round_info["points_difference_on_agreement"] = (points - coplayer_points) if info['round_agreements_reached'][i] else None

        if "cooperative_points_difference_on_agreement" in stats_to_log:
            if agreement_reached:
                coop_player, _ = compute_cooperative_points_for_round(values, coplayer_values, quantities)
                round_info["cooperative_points_difference_on_agreement"] = player_points - coop_player
            else:
                round_info["cooperative_points_difference_on_agreement"] = None

        if "greedy_dominant_points_difference_on_agreement" in stats_to_log:
            if agreement_reached:
                gd_points = compute_greedy_dominant_points(values, quantities)
                round_info["greedy_dominant_points_difference_on_agreement"] = player_points - gd_points
            else:
                round_info["greedy_dominant_points_difference_on_agreement"] = None

        if "greedy_submission_points_difference_on_agreement" in stats_to_log:
            if agreement_reached:
                gs_points = compute_greedy_submission_points(values)
                round_info["greedy_submission_points_difference_on_agreement"] = player_points - gs_points
            else:
                round_info["greedy_submission_points_difference_on_agreement"] = None

        if "split_equal_points_difference_on_agreement" in stats_to_log:
            if agreement_reached:
                se_points = compute_split_equal_points(values, quantities)
                round_info["split_equal_points_difference_on_agreement"] = player_points - se_points
            else:
                round_info["split_equal_points_difference_on_agreement"] = None

        statistics[f"round_{i}"] = round_info

    if "total_coop_points_difference_on_agreement_player" in stats_to_log:
        statistics["total_coop_points_difference_on_agreement_player"] = total_points_player - total_coop_player

    if "total_coop_points_difference_on_agreement_coplayer" in stats_to_log:
        statistics["total_coop_points_difference_on_agreement_coplayer"] = total_points_coplayer - total_coop_coplayer

    if "total_imbalance_on_agreement" in stats_to_log:
        statistics["total_imbalance_on_agreement"] = float(abs(total_points_player - total_points_coplayer) / (total_points_player + total_points_coplayer + 1e-6))

    if "total_points_difference_on_agreement" in stats_to_log:
        statistics["total_points_difference_on_agreement"] = float(total_points_player - total_points_coplayer)
    
    # --- NEW: Statistics computed over all rounds (i.e. independent of agreement) ---
    if "total_coop_points_difference_player" in stats_to_log:
        statistics["total_coop_points_difference_player"] = total_points_player_all - total_coop_player_all

    if "total_coop_points_difference_coplayer" in stats_to_log:
        statistics["total_coop_points_difference_coplayer"] = total_points_coplayer_all - total_coop_coplayer_all

    return statistics


def calculate_imbalance(points, player_role, other_role):
    """
    Calculates the imbalance between the points of the player and the other player.
    
    Args:
        points (dict): A dictionary containing points for each role.
        player_role (str): The role of the player.
        other_role (str): The role of the other player.
    
    scores:
        float: The calculated imbalance.
    """
    total_points = points[player_role] + points[other_role]
    if total_points == 0:
        return 0
    return abs((points[player_role] - points[other_role]) / total_points)


def calculate_items_given_to_self(finalization):
    """
    Calculates the total items (or value thereof) given to self from finalization data.
    
    Args:
        finalization (dict): A dictionary with values representing item amounts.
    
    scores:
        float or None: The sum of the values if valid; otherwise, None.
    """
    if not finalization or not all(isinstance(x, (int, float)) for x in finalization.values()):
        return None
    return sum(finalization.values())


# --- External points computation functions --- #

def compute_points_difference(player_points, other_points):
    """
    Computes the difference between the player's points and the other player's points.
    
    Args:
        player_points (numeric): Points for the player.
        other_points (numeric): Points for the other player.
    
    scores:
        numeric: The difference (player_points - other_points).
    """
    return player_points - other_points


def compute_points_on_agreement(player_points, agreement_reached):
    """
    scores the player's points if an agreement was reached, otherwise None.
    
    Args:
        player_points (numeric): The player's points.
        agreement_reached (bool): Whether an agreement was reached.
    
    scores:
        numeric or None
    """
    return player_points if agreement_reached else None


def compute_points_diff_on_agreement(player_points, other_points, agreement_reached):
    """
    Computes the difference between the player's and the other player's points
    only if an agreement was reached.
    
    Args:
        player_points (numeric): The player's points.
        other_points (numeric): The other player's points.
        agreement_reached (bool): Whether an agreement was reached.
    
    scores:
        numeric or None: The difference if agreement reached; otherwise, None.
    """
    return player_points - other_points if agreement_reached else None


def compute_cooperative_points_for_round(player_values, coplayer_values, quantities):
    """
    Computes the hypothetical cooperative policy points for both players for a round.
    The rule used here is symmetric:
      - If player_values[item] > coplayer_values[item]:
          the player receives full credit (value * quantity) and vice versa.
      - If the values are equal, each receives half credit.
    
    Args:
        player_values (dict): The player's values for each item.
        coplayer_values (dict): The co-player's values for each item.
        quantities (dict): The quantities for each item.
    
    scores:
        tuple: (player_cooperative_points, coplayer_cooperative_points)
    """
    player_coop = 0
    coplayer_coop = 0
    for item in quantities:
        if player_values[item] > coplayer_values[item]:
            player_coop += player_values[item] * quantities[item]
        elif player_values[item] < coplayer_values[item]:
            coplayer_coop += coplayer_values[item] * quantities[item]
        else:
            player_coop += 0.5 * player_values[item] * quantities[item]
            coplayer_coop += 0.5 * coplayer_values[item] * quantities[item]
    return player_coop, coplayer_coop


def compute_greedy_dominant_points(player_values, quantities):
    """
    Computes the hypothetical points under a greedy-dominant policy for the player.
    Policy: Summing the (value * quantity) for all items then subtracting the minimum value.
    
    Args:
        player_values (dict): The player's values for each item.
        quantities (dict): The quantities for each item.
    
    scores:
        numeric: The computed greedy dominant points.
    """
    total = sum(player_values[item] * quantities[item] for item in quantities)
    return total - min(player_values.values()) if player_values else total


def compute_greedy_submission_points(player_values):
    """
    Computes the hypothetical points under a greedy-submission policy by taking the minimum value.
    
    Args:
        player_values (dict): The player's values for each item.
    
    scores:
        numeric: The minimum value among the player's item values.
    """
    return min(player_values.values()) if player_values else None


def compute_split_equal_points(player_values, quantities):
    """
    Computes the hypothetical points under a split-equal policy.
    Policy: Each item contributes half of its full (value * quantity).
    
    Args:
        player_values (dict): The player's values for each item.
        quantities (dict): The quantities for each item.
    
    scores:
        numeric: The computed split-equal points.
    """
    return sum(0.5 * player_values[item] * quantities[item] for item in quantities)