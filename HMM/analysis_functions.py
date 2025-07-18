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

    
def compute_time_in_states(states_sequence, mouse_frames, model):

    states_id = np.arange(len(model.transmat_))

    time_in_states = [0]*len(states_id)

    for i,state in enumerate(states_sequence):

        dt = mouse_frames[i][1] - mouse_frames[i][0]

        time_in_states[state] += dt
    
    return np.array(time_in_states)

def compute_time_in_states_persession(path_to_data_folder, mouse_name, sessions_index, model):

    states_time_distributions = np.array([[]]*6)
    states_time_ratio_distributions = np.array([[]]*6)

    for session_index in sessions_index:
    
        mouse_actions_sequence, runs_frames, _ = extract_actions_sequence(path_to_data_folder, mouse_name, session_index)
        
        states_sequence = model.predict(np.int16(mouse_actions_sequence.reshape(-1,1)))
        # states_sequences.append(states_sequence)

        time_in_states = compute_time_in_states(states_sequence, runs_frames, model)

        ratio_in_states = time_in_states/np.sum(time_in_states)

        # print(states_distribution)

        states_time_distributions = np.append(states_time_distributions,time_in_states.reshape(-1,1),axis=1)
        states_time_ratio_distributions = np.append(states_time_ratio_distributions,ratio_in_states.reshape(-1,1),axis=1)

        # states_distributions = np.append(states_distributions,[states_distribution], axis=0) if len(states_distributions)!=1 else np.array([states_distributions])

    # print(states_distributions)

    return states_time_distributions, states_time_ratio_distributions