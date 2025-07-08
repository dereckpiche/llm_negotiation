from mllm.training.reinforce_trainer import ReinforceTrainerWRS
import matplotlib.pyplot as plt
import numpy as np

ad_align_weights = ReinforceTrainerWRS.get_advantage_alignment_weights(
    advantages=np.random.randn(1, 10), beta=1.0, gamma=0.9
)
import pdb

pdb.set_trace()
