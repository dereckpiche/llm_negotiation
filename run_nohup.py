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

if __name__ == "__main__":
    # Example usage
    command_list = [
        "source /home/mila/d/dereck.piche/negenv/bin/activate", 
        # "/home/mila/d/dereck.piche/negenv/bin/python3.10 /home/mila/d/dereck.piche/llm_negotiation/src/run.py matches.dond_game_args.max_turns=6",
        # "/home/mila/d/dereck.piche/negenv/bin/python3.10 /home/mila/d/dereck.piche/llm_negotiation/src/run.py matches.dond_game_args.rounds_per_game=16",
        "/home/mila/d/dereck.piche/negenv/bin/python3.10 /home/mila/d/dereck.piche/llm_negotiation/src/run.py models.llama.init_args.eval_with=hf models.llama.init_args.keep_hf_during_eval=True"
    ]
    run_commands_with_nohup(command_list)
