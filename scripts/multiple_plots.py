import json

import matplotlib.pyplot as plt
import numpy as np

plot_name = "gpt4o_vs_qwen_items_given_to_self"
# TITLE = "Percentage of Items Allocated to Self"
X_AXIS = "Training Steps"
Y_AXIS = "Percentage of Coins Allocated"

filepaths = {
    "Frozen GPT-4o": "/home/mila/u/udaykaran.kapur/scratch/llm_negotiation/gpt4o_vs_qwen/seed_3/0_statistics/Alice/items_given_to_self_percentage.json",
    "Qwen2.5-7B-Instruct (LoRA fine-tuned)": "/home/mila/u/udaykaran.kapur/scratch/llm_negotiation/gpt4o_vs_qwen/seed_3/0_statistics/Bob/items_given_to_self_percentage.json",
}

colors = {
    "Frozen GPT-4o": "#730106",  # DarkGreen
    "Qwen2.5-7B-Instruct (LoRA fine-tuned)": "#006400",  # Blue
}

data = {}

for agent, file_name in filepaths.items():
    with open(file_name, "r") as f:
        if file_name.endswith(".json"):
            s = json.load(f)
        elif file_name.endswith(".npy"):
            s = np.load(file_name)
        elif file_name.endswith(".csv"):
            s = np.genfromtxt(file_name, delimiter=",")
        data[agent] = np.array(s)

plt.figure(figsize=(8, 5.7))  # Wider figure for legend
plt.grid(True)

for agent, instance in data.items():
    plt.plot(instance, label=agent, color=colors[agent], linewidth=3)

plt.xlabel(X_AXIS, fontsize=14)
plt.ylabel(Y_AXIS, fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# 👉 Legend to the right of the plot
plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=11.5)

plt.tight_layout(rect=[0, 0, 0.85, 1])  # Make space on right for legend
plt.savefig(f"{plot_name}.pdf", format="pdf", bbox_inches="tight", dpi=300)
plt.close()
