# local imports
from utils.common_imports import *
from environments.dond.dond_log_funcs import *

def run_matches(
              matches,
              models,
              log_func,
              log_func_args,
              export_path,
              nb_parallel_matches):
    """
    Runs multiple games in parallel and logs the results.

    Args:
        matches (list): List of match dictionaries.
        models (dict): Dictionary of models to use for generating player moves.
        export_folder (str): Base folder to save player contexts.
        nb_parallel_matches (int): Number of matches to run in parallel.
        game_json_path (str): Path to save game metrics.
        log_matches (bool): Whether to log matches after completion.

    Returns:
        None
    """
    # TODO: Ask Dereck to explain this code (I don't understand it), where are deciding if output is error and requering a new match?
    if nb_parallel_matches == -1:
        nb_parallel_matches = len(matches)

    # Use the provided list of match dictionaries directly (no deep copy)
    all_matches = matches
    parallel_matches = [all_matches.pop(0) for _ in range(min(nb_parallel_matches, len(all_matches)))]

    # Get all the adapter names used by the players
    mod_adpt_ids = []
    for match in parallel_matches:
        for player in match["players"].values():
            if player.mod_adpt_id not in mod_adpt_ids:
                mod_adpt_ids.append(player.mod_adpt_id)
    prompt_batches = {mod_adpt_id: [] for mod_adpt_id in mod_adpt_ids}
    response_batches = {mod_adpt_id: [] for mod_adpt_id in mod_adpt_ids}

    while parallel_matches or all_matches:

        # Build prompt batches for each model
        for match in parallel_matches:
            match["game_state"] = match["game"].get_state()
            current_player = match["players"][match["game"].get_current_player()]
            current_player.set_usr_message(match["game_state"])
            prompt_batches[current_player.mod_adpt_id].append(
                current_player.get_chat_history()  # No deep copy needed here
            )

        # Process prompts for each model
        for mod_adpt_id in mod_adpt_ids:
            model_name = mod_adpt_id.split("/")[0]
            adapter_name = mod_adpt_id.split("/")[1]
            model = models[model_name]
            if prompt_batches[mod_adpt_id]:
                if hasattr(model, 'adapter_names'):
                    model.prepare_adapter_eval(adapter_name)
                response_batches[mod_adpt_id] = model.prompt(prompt_batches[mod_adpt_id])
            prompt_batches[mod_adpt_id] = []

        # Execute player moves based on responses
        for match in parallel_matches[:]:
            match["game_state"] = match["game"].get_state()
            current_player = match["players"][match["game"].get_current_player()]
            response = response_batches[current_player.mod_adpt_id].pop(0)

            action, player_state, send_to_game, player_info = current_player.step(
                input=(match["game_state"], match["game"].get_info(), response)
            )

            if send_to_game:
                observation, reward, done, info = match["game"].step(action)
                match["game_state"] = observation

                if done:
                    # Log game results
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




