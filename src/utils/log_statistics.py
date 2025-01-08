import json
import os
from statistics import mean
from utils.augmented_mean import augmented_mean
from utils.augmented_variance import augmented_variance
from utils.plot_curves import plot_curves
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
from torch.utils.tensorboard import SummaryWriter




import json
import os
from statistics import mean
from utils.augmented_mean import augmented_mean
from utils.augmented_variance import augmented_variance
from utils.plot_curves import plot_curves
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
from torch.utils.tensorboard import SummaryWriter
from utils.statree_utilities import *

def update_player_statistics(input_path, output_file):
    """
    Computes statistics for the current iteration and updates the global statistics file.
    
    Args:
        input_path (str): Path to the folder containing player JSON files for the current iteration.
        output_file (str): Path to the JSON file where statistics are stored.
    """

    # Build statree by appending each dict from JSON files in "input_path" folder
    statree = {}
    for filename in os.listdir(input_path):
        if filename.endswith('.json'):
            with open(os.path.join(input_path, filename), 'r') as f:
                data = json.load(f)
                append_statree(statree, data)
    # Get epoch mean statree
    mean_statree = get_mean_statree(statree)

    # Add mean statree to global stats file
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            global_stats = json.load(f)
    else:
        global_stats = {}

    append_statree(global_stats, mean_statree)

    with open(output_file, 'w') as f:
        json.dump(global_stats, f, indent=4)

    
def generate_player_stats_plots(global_stats_path, plot_folder, tensorboard_log_dir):
    """
    Visualizes the global statistics by generating plots and logging to TensorBoard.

    Args:
        global_stats_path (str): Path to the global statistics JSON file.
        plot_folder (str): Folder to save the plots.
        tensorboard_log_dir (str): Directory to save TensorBoard logs.
    """
    with open(global_stats_path, 'r') as f:
        global_stats = json.load(f)

    # Plot statistics and save to folder
    plot_statree(global_stats, plot_folder)

    # Log statistics to TensorBoard
    writer = SummaryWriter(tensorboard_log_dir)
    tb_statree(global_stats, writer)
    writer.close()

