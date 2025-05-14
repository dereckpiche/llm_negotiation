import json

import matplotlib.pyplot as plt
import numpy as np

mapping = {
    "09-30-18_seed_33": "33",
    "09-30-35_seed_53": "53",
    "09-33-36_seed_6368": "6368",
}

# BASE_PATH = "/home/mila/u/udaykaran.kapur/scratch/outputs/2025-04-19"
BASE_PATH = "/home/mila/u/udaykaran.kapur/scratch/outputs/qwen_ultimatum_game_3e6"

data = []

for folder, seed in mapping.items():
    file_name = (
        f"{BASE_PATH}/{folder}/seed_{seed}/0_statistics/Alice/points_on_agreement.json"
    )
    with open(file_name, "r") as f:
        s = json.load(f)
        data.append(s)

metric_data = np.array(data)

metric_mean = np.mean(metric_data, axis=0)
metric_std = np.std(metric_data, axis=0)

plt.figure()

for instance in metric_data:
    plt.plot(instance, color="lightblue", alpha=0.5)

# Overlay mean curve with dark blue
plt.plot(metric_mean, linewidth=2, color="darkblue", label=f"Average")

# Formatting and saving the plot
plt.title("Seed Averaged Points on Agreement")

plt.xlabel("Iterations")
plt.ylabel("Points")
plt.legend()

plt.savefig("seed_average_points_on_agreement.png", bbox_inches="tight")
plt.close()
