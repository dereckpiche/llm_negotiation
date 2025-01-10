import subprocess

def run_commands_with_nohup(commands, output_file="nohup.out"):
    """
    Runs a list of Linux commands in the background using 'nohup' to ensure they persist after the terminal is closed.

    :param commands: List of command strings to execute.
    :param output_file: File to redirect the nohup output (default: nohup.out).
    """
    try:
        # Loop through each command and execute it with nohup
        for command in commands:
            nohup_command = f"nohup {command} > {output_file} 2>&1 &"
            print(f"Executing: {nohup_command}")
            subprocess.run(nohup_command, shell=True, check=True)
        print("All commands are running in the background with nohup.")
    except subprocess.CalledProcessError as e:
        print(f"Error while executing commands: {e}")
    except Exception as ex:
        print(f"An unexpected error occurred: {ex}")



pyenv = "source /home/mila/d/dereck.piche/negenv/bin/activate"

six_messages = "/home/mila/d/dereck.piche/negenv/bin/python3.10 /home/mila/d/dereck.piche/llm_negotiation/src/run.py matches.dond_game_args.max_turns=6"

sixteen_rounds = "/home/mila/d/dereck.piche/negenv/bin/python3.10 /home/mila/d/dereck.piche/llm_negotiation/src/run.py matches.dond_game_args.rounds_per_game=16"

hf_inference = "/home/mila/d/dereck.piche/negenv/bin/python3.10 /home/mila/d/dereck.piche/llm_negotiation/src/run.py models.llama.init_args.eval_with=hf models.llama.init_args.keep_hf_during_eval=True"

adv_align_test = "/home/mila/d/dereck.piche/negenv/bin/python3.10 /home/mila/d/dereck.piche/llm_negotiation/src/run.py matches.dond_game_args.rounds_per_game=3 matches.run_matches_args.log_func_args.training_data_func=set_discounted_advalign_returns"

if __name__ == "__main__":
    # Example usage
    command_list = [

    ]
    run_commands_with_nohup(command_list)


# Command list archive
