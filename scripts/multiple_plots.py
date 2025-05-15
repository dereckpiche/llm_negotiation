import json

import matplotlib.pyplot as plt
import numpy as np

plot_name = "test_plot"
TITLE = "Percentage of Items Allocated to Self"
X_AXIS = "Training Steps"
Y_AXIS = "Percentage of Items Allocated"

filepaths = {
    "Frozen GPT-4o": "/home/mila/u/udaykaran.kapur/scratch/llm_negotiation/gpt4o_vs_qwen/seed_3/0_statistics/Alice/items_given_to_self_percentage.json",
    "Qwen2.5-7B-Instruct (LoRA fine-tuned)": "/home/mila/u/udaykaran.kapur/scratch/llm_negotiation/gpt4o_vs_qwen/seed_3/0_statistics/Bob/items_given_to_self_percentage.json",
}

data = {}

for agent, file_name in filepaths.items():
    with open(file_name, "r") as f:
        # Handle different file formats
        if file_name.endswith(".json"):
            s = json.load(f)

        elif file_name.endswith(".npy"):
            s = np.load(file_name)

        elif file_name.endswith(".csv"):
            s = np.genfromtxt(file_name, delimiter=",")

        data[agent] = np.array(s)

plt.figure()
# colors = []

for agent, instance in data.items():
    plt.plot(instance, label=agent)

# Formatting and saving the plot
plt.title(TITLE)
plt.xlabel(X_AXIS)
plt.ylabel(Y_AXIS)
plt.legend()
plt.savefig(f"{plot_name}.pdf", format="pdf", bbox_inches="tight")
plt.close()
