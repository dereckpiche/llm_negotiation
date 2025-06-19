import json
import os
import shutil
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from mllm.utils.log_statistics import (
    append_leafstats,
    get_mean_leafstats,
    update_agent_statistics,
)

path = "/home/mila/u/udaykaran.kapur/scratch/outputs/2025-04-11/21-07-57/seed_500/"

it_folders = sorted(
    [
        f
        for f in os.listdir(path)
        if os.path.isdir(os.path.join(path, f)) and f.startswith("iteration_")
    ],
    key=lambda x: int(x.split("_")[1]),
)

agents = ["alice", "bob"]

if os.path.exists("temp/"):
    shutil.rmtree("temp/")

os.makedirs("temp/", exist_ok=True)

for it_folder in it_folders:
    print(it_folder)

    for agent_name in agents:
        raw_data_path = os.path.join(path, it_folder, agent_name, "raw_data")
        stat_path = os.path.join(path, it_folder, agent_name, "statistics")

        alice_first_file_list = []
        bob_first_file_list = []

        for i in os.listdir(raw_data_path):
            file = os.path.join(raw_data_path, i)

            with open(file, "r") as raw_data:
                data = json.load(raw_data)

            if (
                data[0]["role"] == "user"
                and data[0]["round_nb"] == 0
                and "You are the starting agent" in data[0]["content"]
            ):
                stat_file = i.replace("match_", "metrics_")
                if agent_name == "alice":
                    alice_first_file_list.append(stat_file)
                else:
                    bob_first_file_list.append(stat_file)

        if agent_name == "alice":
            assert (
                len(alice_first_file_list) == 8
            ), "Games that alice starts are incorrect."
            first_stat_files = [
                os.path.join(stat_path, f)
                for f in os.listdir(stat_path)
                if f in alice_first_file_list
            ]
            second_stat_files = [
                os.path.join(stat_path, f)
                for f in os.listdir(stat_path)
                if f not in alice_first_file_list
            ]
        else:
            assert len(bob_first_file_list) == 8, "Games that bob starts are incorrect."
            first_stat_files = [
                os.path.join(stat_path, f)
                for f in os.listdir(stat_path)
                if f in bob_first_file_list
            ]
            second_stat_files = [
                os.path.join(stat_path, f)
                for f in os.listdir(stat_path)
                if f not in bob_first_file_list
            ]

        os.makedirs(f"temp/{agent_name}_first", exist_ok=True)
        os.makedirs(f"temp/{agent_name}_second", exist_ok=True)

        first_output_file = os.path.join(
            f"temp/{agent_name}_first", f"{agent_name}_first_stats.json"
        )
        second_output_file = os.path.join(
            f"temp/{agent_name}_second", f"{agent_name}_second_stats.json"
        )

        # Build leafstats by appending each dict from JSON files in "input_path" folder
        leafstats = {}

        for filename in first_stat_files:
            if filename.endswith(".json"):
                with open(filename, "r") as f:
                    data = json.load(f)
                    append_leafstats(leafstats, data)

        # Get epoch mean leafstats
        mean_leafstats = get_mean_leafstats(leafstats)

        # Add mean leafstats to global stats file
        if os.path.exists(first_output_file):
            with open(first_output_file, "r") as f:
                global_stats = json.load(f)
        else:
            global_stats = {}

        append_leafstats(global_stats, mean_leafstats)

        with open(first_output_file, "w") as f:
            json.dump(global_stats, f, indent=4)

        # Build leafstats by appending each dict from JSON files in "input_path" folder
        leafstats = {}

        for filename in second_stat_files:
            if filename.endswith(".json"):
                with open(filename, "r") as f:
                    data = json.load(f)
                    append_leafstats(leafstats, data)

        # Get epoch mean leafstats
        mean_leafstats = get_mean_leafstats(leafstats)

        # Add mean leafstats to global stats file
        if os.path.exists(second_output_file):
            with open(second_output_file, "r") as f:
                global_stats = json.load(f)
        else:
            global_stats = {}

        append_leafstats(global_stats, mean_leafstats)

        with open(second_output_file, "w") as f:
            json.dump(global_stats, f, indent=4)
