"""
This file contains the methods used to convert raw data into training data
for the Negotiation game. (Also called Deal or No Deal).
"""

from utils.common_imports import *


def ipd_generate_training_data_from_raw(
    raw_data_folder:str,
    training_data_folder:str,
    exclude_errors:bool,
    normalize_round_points:bool
):
    """
    Generates training data from raw match data by calculating scores.

    Args:
        raw_data_folder (str):
            Path to the folder containing raw match data.
        training_data_folder (str):
            Path to save the processed training data.
        discount_factor (float):
            The discount factor to apply to future scores.
        exclude_errors (bool):
            If True, exclude messages with "is_error" set to True.
        score_normalize_func (callable, optional):
            Function that takes a list of raw scores and returns a new list
            of shaped scores.
    """

    # Find the score of each round of each game of agent associated with "raw_data_folder"
    round_points_agent, round_points_coagent = ipd_get_round_points_arrays(
        raw_data_folder
    )


    os.makedirs(training_data_folder, exist_ok=True)
    debug_output_folder = os.path.join(
        os.path.dirname(training_data_folder), "ipd_point_arrays_for_db"
    )
    os.makedirs(debug_output_folder, exist_ok=True)
    # Export round_points_agent, round_points_coagent, and scores as CSV in debug folder
    pd.DataFrame(round_points_agent).to_csv(
        os.path.join(debug_output_folder, "round_points_agent.csv"), index=False
    )
    pd.DataFrame(round_points_coagent).to_csv(
        os.path.join(debug_output_folder, "round_points_coagent.csv"), index=False
    )
    if normalize_round_points == True:

        # Subtracts round-wise mean point baseline 
        def sub_loo_mr(array: np.ndarray):
            n = array.shape[0]
            return array - (np.sum(array, axis=0, keepdims=True) - array) / (n - 1)

        round_points_agent = sub_loo_mr(round_points_agent)
        round_points_coagent = sub_loo_mr(round_points_coagent)

        pd.DataFrame(round_points_agent).to_csv(
            os.path.join(debug_output_folder, "normalized_round_points_agent.csv"), index=False
        )
        pd.DataFrame(round_points_coagent).to_csv(
            os.path.join(debug_output_folder, "normalized_round_points_coagent.csv"), index=False
        )

    # Create training data, giving each action their score
    match_files = [
        f
        for f in os.listdir(raw_data_folder)
        if f.startswith("match_") and f.endswith(".json")
    ]
    match_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    # matches = [json.load(open(os.path.join(raw_data_folder, f), 'r')) for f in match_files]

    if not match_files:
        print(f"No raw data files found in {raw_data_folder}")
        return

    for i, match_file in enumerate(match_files):
        chat_history = json.load(open(os.path.join(raw_data_folder, match_file), "r"))

        if exclude_errors:
            chat_history = [
                msg for msg in chat_history if not msg.get("is_error", False)
            ]
        # Attribute scores to actions
        for message in chat_history:
            if message.get("role") == "assistant":
                round_nb = message.get("round_nb")
                message["reward"] = float(round_points_agent[i, round_nb])
                message["co_reward"] = float(round_points_coagent[i, round_nb])

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


def ipd_get_round_points_arrays(raw_data_folder):
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
        [match[-1].get("game_info").get("number_of_rounds") for match in matches]
    )
    round_points_agent = np.full((len(matches), max_rounds), None)
    round_points_coagent = np.full((len(matches), max_rounds), None)

    for i, match in enumerate(matches):
        game_info = match[-1].get("game_info")
        agent_id = match[-1].get("agent_id")
        coagent_id = next(id for id in game_info.get("agent_ids") if id != agent_id)
        nb_rounds = game_info.get("number_of_rounds")
        for round in range(nb_rounds):
            agent_reward = game_info.get("rewards")[round].get(agent_id)
            coagent_reward = game_info.get("rewards")[round].get(coagent_id)
            round_points_agent[i, round] = agent_reward
            round_points_coagent[i, round] = coagent_reward

    return round_points_agent, round_points_coagent


def get_system_msg(match):
    system_msg = next((d for d in match if d.get("role") == "system"), None)
    return system_msg
