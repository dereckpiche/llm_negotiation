#!/bin/bash

# Define the commands for each pair of goals
unbiased_test='python src/run.py matches.players.alice.dond_player_args.goal_prompt=${prompt_blocks.unbiased_goal} matches.players.bob.dond_player_args.goal_prompt=${prompt_blocks.unbiased_goal} hydra.run.dir=outputs/2025-02-20/19-42-41'

normal_bias_test='python src/run.py matches.players.alice.dond_player_args.goal_prompt=${prompt_blocks.normal_goal} matches.players.bob.dond_player_args.goal_prompt=${prompt_blocks.normal_goal}'

agressive_bias_test='python src/run.py matches.players.alice.dond_player_args.goal_prompt=${prompt_blocks.agressive_goal} matches.players.bob.dond_player_args.goal_prompt=${prompt_blocks.agressive_goal} hydra.run.dir=outputs/2025-02-21/02-25-31'

agressive_pair_bias_test='python src/run.py matches.players.alice.dond_player_args.goal_prompt=${prompt_blocks.agressive_pair_goal} matches.players.bob.dond_player_args.goal_prompt=${prompt_blocks.agressive_pair_goal}'

fair_bias_test='python src/run.py matches.players.alice.dond_player_args.goal_prompt=${prompt_blocks.fair_goal} matches.players.bob.dond_player_args.goal_prompt=${prompt_blocks.fair_goal}'

submitted_goal_test='python src/run.py matches.players.alice.dond_player_args.goal_prompt=${prompt_blocks.normal_goal} matches.players.bob.dond_player_args.goal_prompt=${prompt_blocks.submitted_goal}'

# List of commands to execute
command_list=(
    "$unbiased_test"
    # "$normal_bias_test"
    "$agressive_bias_test"
    # "$agressive_pair_bias_test"
    # "$fair_bias_test"
)

# Execute each command
for command in "${command_list[@]}"; do
    echo "Running command: $command"
    # Check if the command is not empty
    if [ -n "$command" ]; then
        if $command > command_output.log 2>&1; then
            echo "Command completed successfully"
        else
            echo "Error while executing command: $command"
            echo "Check command_output.log for details"
        fi
    else
        echo "Command is empty, skipping execution"
    fi
done