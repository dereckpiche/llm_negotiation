# local imports
from utils.common_imports import *
from environments.dond.dond_log_funcs import *

def run_matches(
              matches,
              models,
              iteration,
              log_func,
              log_func_args,
              export_path,
              nb_parallel_matches
              ):
    """
    Runs multiple games in parallel and logs the results.

    Args:
        matches (list): List of match dictionaries.
        models (dict): Dictionary of models to use for generating player moves.
        iteration (int): Iteration number
        log_func (str): Name of the function to use for logging results.
        log_func_args (dict): Arguments for the log function.
        export_path (str): Base folder to save player contexts.
        nb_parallel_matches (int): Number of matches to run in parallel.

    Returns:
        None
    """
    if nb_parallel_matches == -1:
        nb_parallel_matches = len(matches)

    # Use the provided list of match dictionaries directly (no deep copy)
    all_matches = matches
    parallel_matches = [all_matches.pop(0) for _ in range(min(nb_parallel_matches, len(all_matches)))]

    # Get all the adapter names used by the players
    policy_ids = []
    for match in parallel_matches:
        for player in match["players"].values():
            if player.policy_id not in policy_ids:
                policy_ids.append(player.policy_id)
    prompt_batches = {policy_id: [] for policy_id in policy_ids}
    response_batches = {policy_id: [] for policy_id in policy_ids}

    while parallel_matches or all_matches:

        # Build prompt batches for each model
        for match in parallel_matches:
            match["game_state"] = match["game"].get_state()
            current_player = match["players"][match["game"].get_current_player()]
            current_player.set_usr_message(match["game_state"])
            prompt_batches[current_player.policy_id].append(
                current_player.get_chat_history()  # No deep copy needed here
            )

        # Process prompts for each model
        for policy_id in policy_ids:
            model_name = policy_id.split("/")[0]
            adapter_name = policy_id.split("/")[1]
            model = models[model_name]
            if prompt_batches[policy_id]:
                if hasattr(model, 'adapters'):
                    model.prepare_adapter_eval(adapter_name, iteration)
                response_batches[policy_id] = model.prompt(prompt_batches[policy_id])
            prompt_batches[policy_id] = []

        # Execute player moves based on responses
        for match in parallel_matches[:]:
            match["game_state"] = match["game"].get_state()
            current_player = match["players"][match["game"].get_current_player()]
            response = response_batches[current_player.policy_id].pop(0)

            action, player_state, send_to_game, player_info = current_player.step(
                input=(match["game_state"], match["game"].get_info(), response)
            )

            if send_to_game:
                observation, reward, done, info = match["game"].step(action)
                match["game_state"] = observation

                if done:
                    # Log game data
                    player_infos = []
                    for player in match["players"].values():
                        player_infos.append(player.get_info())
                    globals()[log_func](export_path, player_infos, info, **log_func_args)
                    match["game"].reset()

                    # Remove the completed match
                    parallel_matches.remove(match)

                    # Add a new match if available
                    if all_matches:
                        parallel_matches.append(all_matches.pop(0))

    return None




