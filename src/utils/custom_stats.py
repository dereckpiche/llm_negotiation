import json
import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def plot_cumulative_points(json_path):
    """
    Generates a plot of cumulative points for Alice and Bob over iterations 
    from a JSON file and saves it in the same directory as the input JSON.

    Args:
        json_path (str): Path to the JSON file containing the statistics.
    """
    try:
        with open(json_path, 'r') as file:
            statistics = json.load(file)
    except FileNotFoundError:
        print(f"Error: File not found at {json_path}. Please check the file path and try again.")
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




def animate_histogram(data_matrix, interval=500, speed_changes=None):
    """
    Creates an animation of histograms from a numpy matrix of historical data.

    Parameters:
        data_matrix (np.ndarray): A 2D numpy array where each row represents a metric, and each column a time step.
        interval (int): Base interval between frames in milliseconds.
        speed_changes (list of tuples): List of (time_step, speed_multiplier) tuples to change speed dynamically.
    """
    if not isinstance(data_matrix, np.ndarray) or len(data_matrix.shape) != 2:
        raise ValueError("data_matrix must be a 2D numpy array.")

    num_metrics, num_time_steps = data_matrix.shape

    if speed_changes is None:
        speed_changes = []

    # Sort speed changes by time step
    speed_changes = sorted(speed_changes, key=lambda x: x[0])

    # Set up the figure and axis
    fig, ax = plt.subplots()
    
    # Calculate global max value to set y-axis limit
    y_max = data_matrix.max() * 1.1  # Add a little space above the maximum value

    ax.set_ylim(0, y_max)

    # Compute frame intervals based on speed changes
    intervals = [interval] * num_time_steps
    for i, (time_step, speed_multiplier) in enumerate(speed_changes):
        end_step = speed_changes[i + 1][0] if i + 1 < len(speed_changes) else num_time_steps
        for step in range(time_step, end_step):
            intervals[step] = interval / speed_multiplier

    def update(frame):
        # Clear the axis for fresh drawing
        ax.clear()
        
        # Create the histogram using the current time step column
        x = np.arange(num_metrics)
        heights = data_matrix[:, frame]
        ax.bar(x, heights, color='blue', alpha=0.7, edgecolor='black')
        
        # Update titles and labels
        ax.set_title(f"Histogram at Time Step {frame+1}")
        ax.set_xlabel("Metric")
        ax.set_ylabel("Value")
        ax.set_ylim(0, y_max)  # Keep y-axis constant

    def frame_generator():
        current_frame = 0
        while current_frame < num_time_steps:
            yield current_frame
            plt.pause(intervals[current_frame] / 1000)  # Pause in seconds
            current_frame += 1

    # Create the animation
    ani = FuncAnimation(fig, update, frames=frame_generator, repeat=True, blit=False)

    plt.show()

# Example Usage:
if __name__ == "__main__":
    # Generate some example data (e.g., 3 metrics, 10 time steps)
    np.random.seed(42)
    example_data = np.random.randint(0, 10, size=(3, 10))

    # Run the animation with dynamic speed changes
    animate_histogram(example_data, interval=300, speed_changes=[(3, 0.5), (6, 2)])



