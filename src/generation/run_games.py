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
              nb_parallel_matches,
              seed_offset=None
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

    # Queue of matches to process
    pending_matches = matches.copy()  # No deep copy needed for list of references
    active_matches = {}
    
    # Initial population of active matches
    for i in range(min(nb_parallel_matches, len(pending_matches))):
        match = pending_matches.pop(0)
        match_id = id(match)  # Use object ID as unique identifier
        
        # Initialize match state
        env = match['env']
        initial_observations = env.reset()
        
        # Store match with its state
        active_matches[match_id] = {
            'env': env,
            'agents': match['agents'],
            'observations': initial_observations,
            'pending_actions': {},
            'agent_needs_policy': {}  # Track which agents need policy outputs
        }
    
    # Main simulation loop
    while active_matches or pending_matches:
        # Dictionary to collect policy inputs by policy ID
        policy_inputs_by_id = {}
        # Track which policy inputs correspond to which match and agent
        policy_input_mapping = {}  # (policy_id, input_index) -> (match_id, agent_id)
        
        # Process each active match to collect policy inputs
        for match_id, match_data in active_matches.items():
            env = match_data['env']
            agents = match_data['agents']
            observations = match_data['observations']
            
            # Check each agent to see if they need a policy decision
            for agent_id, agent_state in agents.items():
                # Only process agents that have observations
                if agent_id in observations:
                    # Process agent state to get policy needs
                    policy_id, policy_input, action, ready, info = agent_state.step(
                        observation_from_env=observations[agent_id]
                    )
                    
                    # If agent needs policy output
                    if not ready:
                        # Initialize list for this policy if not exists
                        if policy_id not in policy_inputs_by_id:
                            policy_inputs_by_id[policy_id] = []
                        
                        # Add input to the batch and store mapping
                        input_index = len(policy_inputs_by_id[policy_id])
                        policy_inputs_by_id[policy_id].append(policy_input)
                        policy_input_mapping[(policy_id, input_index)] = (match_id, agent_id)
                        
                        # Mark this agent as waiting for policy
                        match_data['agent_needs_policy'][agent_id] = (policy_id, input_index)
                    else:
                        # Agent has determined an action without needing policy
                        match_data['pending_actions'][agent_id] = action
        
        # Process all policy inputs to get outputs
        policy_outputs_by_id = process_policy_inputs(models, policy_inputs_by_id)
        
        # Distribute policy outputs to the appropriate agents
        for match_id, match_data in active_matches.items():
            agents = match_data['agents']
            observations = match_data['observations']
            
            # Check each agent that was waiting for a policy output
            for agent_id, policy_info in match_data['agent_needs_policy'].items():
                policy_id, input_index = policy_info
                policy_output = policy_outputs_by_id[policy_id][input_index]
                
                # Process the policy output to get an action
                _, _, action, ready, _ = agents[agent_id].step(
                    observation_from_env=observations[agent_id],
                    policy_output=policy_output
                )
                
                if ready:
                    # Add action to pending actions
                    match_data['pending_actions'][agent_id] = action
            
            # Clear the list of agents waiting for policy
            match_data['agent_needs_policy'] = {}
        
        # Step environments forward with collected actions
        completed_matches = []
        for match_id, match_data in active_matches.items():
            env = match_data['env']
            pending_actions = match_data['pending_actions']
            
            # If we have at least one action, step the environment
            if pending_actions:
                # Step the environment with all pending actions
                new_observations, done, info = env.step(pending_actions)
                
                # Update match data
                match_data['observations'] = new_observations
                match_data['pending_actions'] = {}
                
                # Check if match is completed
                if done:
                    # Collect log information from environment and all agents
                    env_info = env.get_log_info()
                    agent_infos = {agent_id: agent.get_log_info() 
                                   for agent_id, agent in match_data['agents'].items()}
                    
                    # Log game data using the specified logging function
                    globals()[log_func](export_path, agent_infos, env_info, **log_func_args)
                    
                    # Mark match for removal
                    completed_matches.append(match_id)
        
        # Remove completed matches and add new ones if available
        for match_id in completed_matches:
            del active_matches[match_id]
            
            # Add a new match if available
            if pending_matches:
                new_match = pending_matches.pop(0)
                new_match_id = id(new_match)
                
                # Initialize new match
                env = new_match['env']
                initial_observations = env.reset()
                
                active_matches[new_match_id] = {
                    'env': env,
                    'agents': new_match['agents'],
                    'observations': initial_observations,
                    'pending_actions': {},
                    'agent_needs_policy': {}
                }
    
    return None


def process_policy_inputs(models, policy_inputs_by_id, seed_offset=0):
    """
    Process batches of inputs for each policy and return the outputs.
    
    Args:
        models (dict): Dictionary of models to use for generating outputs.
        policy_inputs_by_id (dict): Dictionary mapping policy IDs to lists of inputs.
        
    Returns:
        dict: Dictionary mapping policy IDs to lists of outputs.
    """
    policy_outputs_by_id = {}
    
    for policy_id, inputs in policy_inputs_by_id.items():
        if not inputs:
            policy_outputs_by_id[policy_id] = []
            continue
            
        model_name, adapter_name = policy_id.split("/")
        model = models[model_name]
        
        if hasattr(model, 'adapters'):
            model.prepare_adapter_eval(adapter_name, seed_offset)  # Iteration is handled separately
            
        policy_outputs_by_id[policy_id] = model.prompt(inputs)
        
    return policy_outputs_by_id

