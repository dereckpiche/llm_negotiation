import json
import os
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from utils.statrees import *


def gather_ipd_statistics(raw_data_folder: str):
    """
    Gathers statistics from the raw data folder.
    Args:
        raw_data_folder: The path to the folder containing the raw data.
    Returns:
        A dictionary containing the statistics.
    """
    data = []
    statistics = {}
    for file in os.listdir(raw_data_folder):
        if file.endswith(".json"):
            data.append(json.load(open(os.path.join(raw_data_folder, file))))

    for game in data:
        game_log = game[-1]["game_info"]

        pass

    return statistics
