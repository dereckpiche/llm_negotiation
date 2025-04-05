from utils.common_imports import *
from environments.dond.dond_log_funcs import *

def run_batched_matches(
    matches,
    models,
    export_path,
    nb_parallel_matches,
    seed_offset=0
):
    """
    Runs multiple negotiation games in parallel and logs the results.

    Args:
        matches (list): List of match dictionaries, each containing:
            - 'env': Environment instance
            - 'agents': Dictionary mapping agent IDs to AgentState instances
            - 'log_func': Function object to use for logging this match
            - 'log_func_args': Dictionary of arguments for the log_func
        models (dict): Dictionary of models to use for generating outputs.
        iteration (int): Iteration number
        export_path (str): Base folder to save game contexts.
        nb_parallel_matches (int): Number of matches to run in parallel.

    Returns:
        None
    """
    if nb_parallel_matches == -1:
        nb_parallel_matches = len(matches)

    pending_matches = matches.copy()
    active_matches = {}

    # Initial population of active matches
    for i in range(min(nb_parallel_matches, len(pending_matches))):
        match = pending_matches.pop(0)
        match_id = id(match)

        env = match['env']
        initial_observations = env.reset()

        active_matches[match_id] = {
            'env': env,
            'agents': match['agents'],
            'observations': initial_observations,
            'action_required_agents': list(initial_observations.keys()),
            'pending_actions': {},
            'policy_outputs': {agent_id: None for agent_id in match['agents']},
            'log_func': match['log_func'],
            'log_func_args': match['log_func_args']
        }

    policy_inputs = {}  # {policy_id: {match_id: {agent_id: input}}}

    # Main simulation loop
    while active_matches or pending_matches:

        for match_id, match_data in active_matches.items():
            env = match_data['env']
            agents = match_data['agents']
            observations = match_data['observations']
            action_required_agents = set(match_data['action_required_agents'])
            ready_agents = set(match_data["pending_actions"].keys())
            not_ready_agents = action_required_agents - ready_agents

            for agent_id in not_ready_agents:

                agent_state = agents[agent_id]

                policy_id, policy_input, action, ready, info = agent_state.step(
                    observation_from_env=observations[agent_id],
                    policy_output=match_data['policy_outputs'].get(agent_id, None)
                )

                if not ready:
                    if policy_id not in policy_inputs:
                        policy_inputs[policy_id] = {}
                    if match_id not in policy_inputs[policy_id]:
                        policy_inputs[policy_id][match_id] = {}
                    policy_inputs[policy_id][match_id][agent_id] = policy_input
                    # Reset policy output to None since we're requesting a new one
                    match_data['policy_outputs'][agent_id] = None
                else:
                    match_data['pending_actions'][agent_id] = action
                    match_data['policy_outputs'][agent_id] = None

        # Get policy outputs for the agents -- in a batched and efficient way

        policy_outputs = process_policy_inputs(models, policy_inputs, seed_offset=seed_offset)

        for match_id, match_data in active_matches.items():
            if match_id in policy_outputs:  # Add this check to prevent KeyError
                for agent_id, policy_output in policy_outputs[match_id].items():
                    match_data['policy_outputs'][agent_id] = policy_output

        policy_inputs = {}

        # Step environments forward with collected actions - only when all agents are ready
        completed_matches = []

        for match_id, match_data in active_matches.items():

            pending_actions = match_data['pending_actions']
            action_required_agents = set(match_data['action_required_agents'])
            ready_agents = set(pending_actions.keys())

            # Only step when all agents (with action required) are ready

            if action_required_agents == ready_agents:

                # Take step
                env = match_data['env']
                new_observations, done, info = env.step(pending_actions)
                match_data['observations'] = new_observations
                match_data['action_required_agents'] = list(new_observations.keys())
                match_data['pending_actions'] = {}

                # Trajectory has completed
                if done:
                    env_info = env.get_log_info()
                    agent_infos = [agent.get_log_info() for agent in match_data['agents'].values()]

                    # Use the match-specific log function and args
                    match_data['log_func'](export_path, agent_infos, env_info, **match_data['log_func_args'])
                    completed_matches.append(match_id)

        # Remove completed matches and add new ones
        for match_id in completed_matches:
            del active_matches[match_id]

            if pending_matches:
                new_match = pending_matches.pop(0)
                new_match_id = id(new_match)

                env = new_match['env']
                initial_observations = env.reset()

                active_matches[new_match_id] = {
                    'env': env,
                    'agents': new_match['agents'],
                    'observations': initial_observations,
                    'action_required_agents': list(initial_observations.keys()),
                    'pending_actions': {},
                    'policy_outputs': {agent_id: None for agent_id in new_match['agents']},
                    'log_func': new_match['log_func'],
                    'log_func_args': new_match['log_func_args']
                }

    return None


def process_policy_inputs(models, policy_inputs, seed_offset=0):
    """
    Process batches of inputs for each policy and return the outputs.

    Args:
        models (dict): Dictionary of models to use for generating outputs.
        policy_inputs (dict): Nested dictionary {policy_id: {match_id: {agent_id: input}}}
        seed_offset (int, optional): Offset for seeding, defaults to 0

    Returns:
        dict: Nested dictionary {match_id: {agent_id: output}}
    """
    policy_outputs = {}  # {match_id: {agent_id: output}}

    for policy_id, match_dict in policy_inputs.items():
        if not match_dict:
            continue

        model_name, adapter_name = policy_id.split("/")
        model = models[model_name]

        if hasattr(model, 'adapters'):
            model.prepare_adapter_eval(adapter_name, seed_offset)

        # Flatten inputs for batch processing
        flat_inputs = []
        input_mapping = []  # [(match_id, agent_id), ...]
        for match_id, agent_dict in match_dict.items():
            for agent_id, input_data in agent_dict.items():
                flat_inputs.append(input_data)
                input_mapping.append((match_id, agent_id))

        # Get batch outputs
        batch_outputs = model.prompt(flat_inputs, seed_offset)

        # Reconstruct nested structure
        for (match_id, agent_id), output in zip(input_mapping, batch_outputs):
            if match_id not in policy_outputs:
                policy_outputs[match_id] = {}
            policy_outputs[match_id][agent_id] = output

    return policy_outputs