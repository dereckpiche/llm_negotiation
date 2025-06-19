import json
import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


def plot_cumulative_points(json_path):
    """
    Generates a plot of cumulative points for Alice and Bob over iterations
    from a JSON file and saves it in the same directory as the input JSON.

    Args:
        json_path (str): Path to the JSON file containing the statistics.
    """
    try:
        with open(json_path, "r") as file:
            statistics = json.load(file)
    except FileNotFoundError:
        print(
            f"Error: File not found at {json_path}. Please check the file path and try again."
        )
        return

    nb_epochs = len(statistics["round_0"]["self_points"])
    alice_cumu_points = np.zeros(nb_epochs)
    bob_cumu_points = np.zeros(nb_epochs)

    for round_key in statistics.keys():
        alice_cumu_points += np.array(statistics[round_key]["self_points"])
        bob_cumu_points += np.array(statistics[round_key]["other_points"])

    iterations = np.arange(1, nb_epochs + 1)
    output_dir = os.path.dirname(json_path)
    plot_filename = "average_cumulative_points_plot.png"
    output_path = os.path.join(output_dir, plot_filename)

    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, alice_cumu_points, label="Alice's Avg Cumulative Points")
    plt.plot(iterations, bob_cumu_points, label="Bob's Avg Cumulative Points")
    plt.xlabel("Iterations")
    plt.ylabel("Average Cumulative Points")
    plt.title("Average Cumulative Points Over Iterations")
    plt.legend()
    plt.grid()
    plt.savefig(output_path)

    print(f"Plot saved successfully at: {output_path}")


def update_agent_statistics(input_path, output_file):
    """
    Computes statistics for the current iteration and updates the global statistics file.

    Args:
        input_path (str): Path to the folder containing agent JSON files for the current iteration.
        output_file (str): Path to the JSON file where statistics are stored.
    """

    # Build leafstats by appending each dict from JSON files in "input_path" folder
    leafstats = {}
    for filename in os.listdir(input_path):
        # search in alice "statistics" folder

        if filename.endswith(".json"):
            with open(os.path.join(input_path, filename), "r") as f:
                data = json.load(f)
                append_leafstats(leafstats, data)
    # Get epoch mean leafstats
    var_leafstats = get_var_leafstats(leafstats)

    # Add mean leafstats to global stats file
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            global_stats = json.load(f)
    else:
        global_stats = {}

    append_leafstats(global_stats, var_leafstats)

    with open(output_file, "w") as f:
        json.dump(global_stats, f, indent=4)


def animate_items_given_to_self(base_path, output_file="items_given_to_self.mp4"):
    """
    Generates an animated histogram of items given to self over iterations from multiple JSON files.

    Args:
        base_path (str): Path to the base folder containing iteration folders.
        output_file (str): Path to save the animated histogram as an MP4 file.
    """
    stats = {}

    # Iterate through each iteration folder
    for i, iteration_folder in enumerate(sorted(os.listdir(base_path))):
        iteration_path = os.path.join(
            base_path, iteration_folder, "alice", "statistics"
        )
        if os.path.isdir(iteration_path):
            print(f"Processing iteration {i}...")
            stats[i] = {}
            # Iterate through each JSON file in the iteration folder
            for filename in sorted(os.listdir(iteration_path)):
                if filename.endswith(".json"):
                    with open(os.path.join(iteration_path, filename), "r") as file:
                        data = json.load(file)
                        for round_key in data.keys():
                            items_given = data[round_key]["items_given_to_self"]
                            if items_given not in stats[i]:
                                stats[i][items_given] = 0
                            stats[i][items_given] += 1

    # Check if stats is populated correctly
    if not stats:
        print("No data found to animate.")
        return

    # Create an animated histogram
    fig, ax = plt.subplots()

    def update(frame):
        ax.clear()
        if frame in stats:
            iteration_stats = stats[frame]
            items = list(iteration_stats.keys())
            frequencies = list(iteration_stats.values())
            ax.bar(items, frequencies)
            ax.set_title(f"Iteration {frame}")
            ax.set_xlabel("Items Given to Self")
            ax.set_ylabel("Frequency")
            ax.set_ylim(0, max(max(frequencies), 1))  # Ensure y-axis is not zero
        else:
            print(f"No data for frame {frame}")

    ani = animation.FuncAnimation(fig, update, frames=len(stats), repeat=False)
    ani.save(output_file, writer="ffmpeg")
    plt.close(fig)


def animate_items_given_to_self_by_round(
    base_path, output_file="items_given_to_self_by_round.mp4", pause_frames=5
):
    """
    Generates an animated histogram of items given to self over iterations from multiple JSON files,
    with separate subplots for each round. Each frame is repeated to allow for a longer pause.

    Args:
        base_path (str): Path to the base folder containing iteration folders.
        output_file (str): Path to save the animated histogram as an MP4 file.
        pause_frames (int): Number of times each frame is repeated to create a pause effect.
    """
    stats = {round_num: {} for round_num in range(10)}  # Assuming 10 rounds

    # Iterate through each iteration folder
    for i, iteration_folder in enumerate(sorted(os.listdir(base_path))):
        iteration_path = os.path.join(
            base_path, iteration_folder, "alice", "statistics"
        )
        if os.path.isdir(iteration_path):
            # Iterate through each JSON file in the iteration folder
            for filename in sorted(os.listdir(iteration_path)):
                if filename.endswith(".json"):
                    with open(os.path.join(iteration_path, filename), "r") as file:
                        data = json.load(file)
                        for round_key, round_data in data.items():
                            round_num = int(round_key.split("_")[1])
                            items_given = round_data["items_given_to_self"]
                            if not isinstance(items_given, int) or (
                                items_given < 0 or items_given > 10
                            ):
                                items_given = 0
                            if i not in stats[round_num]:
                                stats[round_num][i] = {}
                            if items_given not in stats[round_num][i]:
                                stats[round_num][i][items_given] = 0
                            stats[round_num][i][items_given] += 1

    # Check if stats is populated correctly
    if not any(stats.values()):
        print("No data found to animate.")
        return

    # Create an animated histogram with subplots
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))  # 2 rows, 5 columns for 10 rounds
    axes = axes.flatten()

    # Adjust the space between plots
    fig.subplots_adjust(
        hspace=0.4, wspace=0.4
    )  # Add more space between rows and columns

    def update(frame):
        actual_frame = (
            frame // pause_frames
        )  # Determine the actual frame based on pause_frames
        for round_num, ax in enumerate(axes):
            ax.clear()
            if actual_frame in stats[round_num]:
                iteration_stats = stats[round_num][actual_frame]
                items = list(iteration_stats.keys())
                frequencies = list(iteration_stats.values())
                ax.bar(items, frequencies)
                ax.set_title(f"Round {round_num} - Iteration {actual_frame}")
                ax.set_xlabel("Items Given to Self")
                ax.set_ylabel("Frequency")
                ax.set_ylim(0, max(max(frequencies), 1))  # Ensure y-axis is not zero
            else:
                ax.set_title(f"Round {round_num} - No Data")

    total_frames = max(len(stats[round_num]) for round_num in stats) * pause_frames
    ani = animation.FuncAnimation(fig, update, frames=total_frames, repeat=False)
    ani.save(output_file, writer="ffmpeg")
    plt.close(fig)


if __name__ == "__main__":
    folder = "important_outputs/2025-01-24 naive rl 10 rounds UG/"
    # output_file = "important_outputs/var_stats.json"

    # # Sort the iterations to ensure they are processed in order
    # for iteration in sorted(os.listdir(folder)):
    #     if iteration.startswith("iteration_"):
    #         update_agent_statistics(os.path.join(folder, iteration, "alice", "statistics"), output_file)

    # # create plots of the var_stats.json file
    # with open(output_file, 'r') as f:
    #     var_stats = json.load(f)

    # plot_leafstats(var_stats, folder="important_outputs/plots")

    # Generate animated histogram for items given to self
    animate_items_given_to_self(folder)
    print("done")
