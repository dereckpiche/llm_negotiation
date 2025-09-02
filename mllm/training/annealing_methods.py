import numpy as np


def sigmoid_annealing(step: int, temperature: float) -> float:
    return 2 / (1 + np.exp(-step / temperature)) - 1

