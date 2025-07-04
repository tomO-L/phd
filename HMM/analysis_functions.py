#######################
### Import packages ###
#######################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from scipy.stats import wilcoxon
import os
import pickle
from hmmlearn import hmm, vhmm
from tqdm import tqdm
from actions_functions import *


########################
### Define functions ###
########################

def compute_states_distribution(states_sequence, model):

    states_id = np.arange(len(model.transmat_))

    states_distribution = np.array([np.count_nonzero(states_sequence==i) for i in states_id])/len(states_sequence)

    return states_distribution

def compute_states_distribution_persession(path_to_data_folder, mouse_name, sessions_index, model):

    # states_sequences = []
    states_distributions = np.array([[]]*6)

    for session_index in sessions_index:
    
        mouse_actions_sequence = extract_actions_sequence(path_to_data_folder, mouse_name, session_index)[0]
    
        states_sequence = model.predict(np.int16(mouse_actions_sequence.reshape(-1,1)))
        # states_sequences.append(states_sequence)

        states_distribution = compute_states_distribution(states_sequence,model)

        # print(states_distribution)

        states_distributions = np.append(states_distributions,states_distribution.reshape(-1,1),axis=1)

        # states_distributions = np.append(states_distributions,[states_distribution], axis=0) if len(states_distributions)!=1 else np.array([states_distributions])

    # print(states_distributions)

    return states_distributions

    