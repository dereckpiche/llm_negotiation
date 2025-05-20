import json

import matplotlib.pyplot as plt
import numpy as np

plot_name = "welfare_comparison"
X_AXIS = "Training Steps"
Y_AXIS = "Percentage of Maximum Welfare"
max_steps = 150

muqeeth_max_file_paths = [
    "/home/mila/d/dereck.piche/scratch/llm_negotiation/sum_points_percentage_of_max_seed42.json",
    "/home/mila/d/dereck.piche/scratch/llm_negotiation/sum_points_percentage_of_max_seed344.json",
    "/home/mila/d/dereck.piche/scratch/llm_negotiation/sum_points_percentage_of_max_seed1000.json",
]

milad_max_file_paths = [
    "/home/mila/d/dereck.piche/scratch/llm_negotiation/REPRODUCE/greedy/seed_645/0_statistics/Alice/sum_points_percentage_of_max.json",
    "/home/mila/d/dereck.piche/scratch/llm_negotiation/seed_42_iterations/0_statistics/Alice/sum_points_percentage_of_max.json",
    "/home/mila/d/dereck.piche/scratch/llm_negotiation/seed_1003_iterations/0_statistics/Alice/sum_points_percentage_of_max.json",
]


filepaths = {
    "Naive MARL": milad_max_file_paths,
    "Cooperative MARL": muqeeth_max_file_paths,
}

# Define a color mapping for each agent
color_mapping = {
    "Naive MARL": "#790D12",  # Red
    "Cooperative MARL": "#006400",  # DarkGreen
    # Add more agents and their associated colors here
}

plt.figure()
plt.grid(True)

for agent, file_list in filepaths.items():
    data = []
    for file_name in file_list:
        with open(file_name, "r") as f:
            # Handle different file formats
            if file_name.endswith(".json"):
                s = json.load(f)
            elif file_name.endswith(".npy"):
                s = np.load(file_name)
            elif file_name.endswith(".csv"):
                s = np.genfromtxt(file_name, delimiter=",")
            data.append(np.array(s))

    # Compute the minimum length across all data arrays
    min_length = min(len(arr) for arr in data)
    min_length = min(min_length, max_steps)  # Limit to max_steps

    # Truncate each array to the minimum length
    data = [arr[:min_length] for arr in data]

    # Get the color for this agent
    color = color_mapping.get(agent)

    # Plot each individual line with reduced alpha
    for instance in data:
        plt.plot(instance, color=color, alpha=0.2)

    # Calculate and plot mean
    metric_mean = np.mean(data, axis=0)
    plt.plot(metric_mean, label=agent, color=color, linewidth=3)

# Formatting and saving the plot
plt.xlabel(X_AXIS)
plt.ylabel(Y_AXIS)
plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
plt.savefig(f"{plot_name}.pdf", format="pdf", bbox_inches="tight")
plt.close()
