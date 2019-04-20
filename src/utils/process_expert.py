import numpy as np 
from copy import deepcopy

CONDITION_FULL = 0
CONDITION_NONE = 1
CONDITION_PARTIAL = 2
CONDITION_NOISY = 3
CONDITION_PARTIAL_NOISY = 4

NUM_CONDITION = 5

def process_expert(expert_obs, expert_acs):
    num_demo = expert_obs.shape[0]
    rand_array = np.random.rand(num_demo)
    idx = np.floor(rand_array).astype(np.uint8)

    return idx