from utils.common_imports import *
from environments.dond.dond_log_funcs import *

def run_matches(
    matches,
    models,
    iteration,
    log_func,
    log_func_args,
    export_path,
    nb_parallel_matches,
    seed_offset=0
):
    """
    Runs multiple negotiation games in parallel and logs the results.

    Args:
        matches (list): List of match dictionaries, each containing:
            - 'env': NegotiationEnvironment instance
            - 'agents': Dictionary mapping agent IDs to AgentState instances
        models (dict): Dictionary of models to use for generating outputs.
        iteration (int): Iteration number
        log_func (str): Name of the function to use for logging results.
        log_func_args (dict): Arguments for the log function.
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
            'pending_actions': {},
            'policy_outputs': {agent_id: None for agent_id in match['agents']}
        }
    
    # Main simulation loop
    while active_matches or pending_matches:
        # Collect policy inputs using nested dictionary structure
        policy_inputs = {}  # {policy_id: {match_id: {agent_id: input}}}
        
        for match_id, match_data in active_matches.items():
            env = match_data['env']
            agents = match_data['agents']
            observations = match_data['observations']
            
            for agent_id, agent_state in agents.items():
                if agent_id in observations.keys():
                    # Pass existing policy output if it exists
                    existing_policy_output = match_data['policy_outputs'][agent_id]
                    policy_id, policy_input, action, ready, info = agent_state.step(
                        observation_from_env=observations[agent_id],
                        policy_output=existing_policy_output
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
                        # Clear policy output if action is ready
                        match_data['policy_outputs'][agent_id] = None
        
        # Process policy inputs and get outputs
        policy_outputs = process_policy_inputs(models, policy_inputs, seed_offset=seed_offset)
        
        # Distribute policy outputs to agents
        for match_id, match_data in active_matches.items():
            agents = match_data['agents']
            observations = match_data['observations']
            
            # Update policy outputs for this match
            if match_id in policy_outputs:
                for agent_id, policy_output in policy_outputs[match_id].items():
                    match_data['policy_outputs'][agent_id] = policy_output
                    
                    # Process the new policy output immediately
                    policy_id, policy_input, action, ready, info = agents[agent_id].step(
                        observation_from_env=observations[agent_id],
                        policy_output=policy_output
                    )
                    if ready:
                        match_data['pending_actions'][agent_id] = action
                        match_data['policy_outputs'][agent_id] = None  # Clear if used
        
        # Step environments forward with collected actions
        completed_matches = []
        for match_id, match_data in active_matches.items():
            env = match_data['env']
            pending_actions = match_data['pending_actions']
            
            if pending_actions:
                #import pdb; pdb.set_trace()
                new_observations, done, info = env.step(pending_actions)
                
                match_data['observations'] = new_observations
                match_data['pending_actions'] = {}
                
                if done:
                    env_info = env.get_log_info()
                    agent_infos = [agent.get_log_info() for agent in match_data['agents'].values()]
                    globals()[log_func](export_path, agent_infos, env_info, **log_func_args)
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
                    'pending_actions': {},
                    'policy_outputs': {agent_id: None for agent_id in new_match['agents']}
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
        batch_outputs = model.prompt(flat_inputs)
        
        # Reconstruct nested structure
        for (match_id, agent_id), output in zip(input_mapping, batch_outputs):
            if match_id not in policy_outputs:
                policy_outputs[match_id] = {}
            policy_outputs[match_id][agent_id] = output
    
    return policy_outputs