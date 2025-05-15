import json

import matplotlib.pyplot as plt
import numpy as np

plot_name = "ipd_coop_rate"
X_AXIS = "Training Steps"
Y_AXIS = "Mutual Cooperation Rate"
max_steps = 80


# filepaths = [
#     "/home/mila/d/dereck.piche/scratch/llm_negotiation/ipd_covert/seed_123/0_statistics/mutual_defection_rate.json",
#     "/home/mila/d/dereck.piche/scratch/llm_negotiation/ipd_covert/seed_657/0_statistics/mutual_defection_rate.json",
#     "/home/mila/d/dereck.piche/scratch/llm_negotiation/ipd_covert/seed_934/0_statistics/mutual_defection_rate.json",
# ]

filepaths = [
    "/home/mila/d/dereck.piche/scratch/llm_negotiation/ipd_overt/seed_155/0_statistics/mutual_cooperation_rate.json",
    "/home/mila/d/dereck.piche/scratch/llm_negotiation/ipd_overt/seed_256/0_statistics/mutual_cooperation_rate.json",
    "/home/mila/d/dereck.piche/scratch/llm_negotiation/ipd_overt/seed_856/0_statistics/mutual_cooperation_rate.json",
]

# mapping = {
#     "09-30-18_seed_33": "33",
#     "09-30-35_seed_53": "53",
#     "09-33-36_seed_6368": "6368",
# }

data = []

for file_name in filepaths:
    with open(file_name, "r") as f:
        # Handle different file formats
        if file_name.endswith(".json"):
            s = json.load(f)
            data.append(np.array(s))
        elif file_name.endswith(".npy"):
            s = np.load(file_name)
            data.append(np.array(s))
        elif file_name.endswith(".csv"):
            s = np.genfromtxt(file_name, delimiter=",")
            data.append(np.array(s))

min_length = min(len(arr) for arr in data)
min_length = min(min_length, max_steps)
data = [arr[:min_length] for arr in data]

metric_data = np.array(data)

metric_mean = np.mean(metric_data, axis=0)
metric_std = np.std(metric_data, axis=0)

plt.figure()
plt.grid(True)

for instance in metric_data:
    plt.plot(
        instance, color="#556B2F", alpha=0.2
    )  # DarkOliveGreen with reduced visibility

# Overlay mean curve with a dark green and no dots
plt.plot(metric_mean, linewidth=3, color="#006400", label=f"Average")  # DarkGreen

# Formatting and saving the plot
plt.xlabel(X_AXIS, fontsize=14)
plt.ylabel(Y_AXIS, fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig(f"{plot_name}.pdf", format="pdf", bbox_inches="tight")
plt.close()
