from utils.common_imports import *


def gather_dond_statistics(player_info, info, stats_to_log):
    """
    Gathers specified statistics of a game for a single player and outputs them in JSONL format.

    Args:
        player_info (dict): A dictionary containing player information.
        info (dict): A dictionary containing game information.
        stats_to_log (list): A list of statistics names to log.

    Returns:
        str: A JSONL string containing the specified game statistics.
    """
    statistics = {}
    player_name = player_info['player_name']

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
        coplayer_points = info['round_points'][i][other_role]

        if "agreement_percentage" in stats_to_log:
            round_info["agreement_percentage"] = 100 if info['round_agreements_reached'][i] else 0

        if "points" in stats_to_log:
            round_info["points"] = info['round_points'][i][player_role]

        if "other_points" in stats_to_log:
            round_info["other_points"] = info['round_points'][i][other_role]

        if "points_difference" in stats_to_log:
            round_info["points_difference"] = info['round_points'][i][player_role] - info['round_points'][i][other_role]

        if "imbalance" in stats_to_log:
            round_info["imbalance"] = calculate_imbalance(info['round_points'][i], player_role, other_role)

        if "items_given_to_self" in stats_to_log:
            round_info["items_given_to_self"] = calculate_items_given_to_self(info['round_finalizations'][i][player_role])

        if "points_on_agreement" in stats_to_log:
            round_info["points_on_agreement"] = info['round_points'][i][player_role] if info['round_agreements_reached'][i] else None

        if "other_points_on_agreement" in stats_to_log:
            round_info["other_points_on_agreement"] = info['round_points'][i][other_role] if info['round_agreements_reached'][i] else None

        if "points_diff_on_agreement" in stats_to_log:
            round_info["points_diff_on_agreement"] = (info['round_points'][i][player_role] - info['round_points'][i][other_role]) if info['round_agreements_reached'][i] else None

        if "quantities" in stats_to_log:
            round_info["quantities"] = quantities

        if "values" in stats_to_log:
            round_info["values"] = values


        if "cooperative_points_percentage" in stats_to_log:
            if info['round_agreements_reached'][i]:
                cooperative_points = 0
                for item in quantities.keys():
                    if values[item] >= coplayer_values[item]:
                        cooperative_points += values[item] * quantities[item]
                    elif values[item] == coplayer_values[item]:
                        cooperative_points += values[item] * quantities[item] / 2
                round_info["cooperative_points_percentage"] = 100 * points / cooperative_points
            else:
                round_info["cooperative_points_percentage"] = None

        if "greedy_dominant_points_percentage" in stats_to_log:
            if info['round_agreements_reached'][i]:
                greedy_dominant_points = 0
                for item in quantities.keys():
                    greedy_dominant_points += values[item] * quantities[item]
                greedy_dominant_points -= min(values.values())
                round_info["greedy_dominant_points_percentage"] = 100 * points / greedy_dominant_points
            else:
                round_info["greedy_dominant_points_percentage"] = None

        if "greedy_submission_points_percentage" in stats_to_log:
            if info['round_agreements_reached'][i]:
                greedy_submission_points = min(values.values())
                round_info["greedy_submission_points_percentage"] = 100 * points / greedy_submission_points
            else:
                round_info["greedy_submission_points_percentage"] = None

        if "split_equal_points_percentage" in stats_to_log:
            if info['round_agreements_reached'][i]:
                split_equal_points = 0
                for item in quantities.keys():
                    split_equal_points += (1/2) * values[item] * quantities[item]
                round_info["split_equal_points_percentage"] = 100 * points / split_equal_points
            else:
                round_info["split_equal_points_percentage"] = None

        statistics[f"round_{i}"] = round_info

    return statistics

def calculate_imbalance(points, player_role, other_role):
    """
    Calculates the imbalance between the points of the player and the other player.

    Args:
        points (dict): A dictionary containing points for each role.
        player_role (str): The role of the player.
        other_role (str): The role of the other player.

    Returns:
        float: The calculated imbalance.
    """
    total_points = points[player_role] + points[other_role]
    if total_points == 0:
        return 0
    return abs((points[player_role] - points[other_role]) / total_points)

def calculate_items_given_to_self(finalization):
    if not finalization or not all(isinstance(x, (int, float)) for x in finalization.values()):
        return None
    return sum(finalization.values())