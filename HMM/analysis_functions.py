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
    l = len(model.transmat_)
    states_distributions = np.array([[]]*l)

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

    l = len(model.transmat_)

    states_time_distributions = np.array([[]]*l)
    states_time_ratio_distributions = np.array([[]]*l)

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


def compute_cumulated_turns_profile(ordered_runs):
    
    """
    Computes the cumulative number of runs around a tower at each time a new run around a tower occurs, for a given session and a given mouse.

    """

    # Initialize lists to store filtered turns
    rank_of_rewarded_qts = []
    rank_of_unrewarded_qts = []

    # Iterate through each recorded turn around the tower
    
    for i in range(len(ordered_runs)):

        if ordered_runs[i][0]=='run_around_tower':

            is_rewarded = ordered_runs[i][4]['Rewarded']

            if is_rewarded:

                rank_of_rewarded_qts.append(i)

            else:

                rank_of_unrewarded_qts.append(i)

    # # Sort the turn times for cumulative calculations
    # CW_times_sorted = np.sort(time_of_runsaroundtower_cw)
    # CCW_times_sorted = np.sort(time_of_runsaroundtower_ccw)

    # # Compute cumulative counts for each direction
    # CW_cumulative = np.arange(1, len(CW_times_sorted) + 1)
    # CCW_cumulative = np.arange(1, len(CCW_times_sorted) + 1)
    

    # good_turns_time = CW_times_sorted
    # bad_turns_time = CCW_times_sorted
    # cumulated_good_turns = CW_cumulative
    # cumulated_bad_turns = CCW_cumulative


    # return [good_turns_time, cumulated_good_turns], [bad_turns_time, cumulated_bad_turns]
    return rank_of_rewarded_qts, rank_of_unrewarded_qts

def compute_cumulated_states_profile(states_sequence, states):

    cumulated_states_steps = []
    
    for s_type in states:

        arr = np.where(states_sequence==s_type)[0]
        cumulated_states_steps.append(arr)

    return cumulated_states_steps

def compute_slope(sequence, window_size=50, weight=1):

    cumulated_quantity = np.cumsum(sequence)

    slope_list = np.diff(cumulated_quantity)

    smoothed_slope_list = np.convolve(slope_list,np.ones(window_size)*weight, mode='same')/window_size
    
    return smoothed_slope_list

def compute_occurence_frequency_v1(sequence,window_size=51):

    occurence_frequency = np.convolve(sequence,np.ones(window_size), mode='same')/window_size
    
    return occurence_frequency

def compute_occurence_frequency_v2(sequence,window_size=51):

    occurence_frequency = []

    for i in range(len(sequence)):

        effective_window_size = window_size if 1>=window_size else i
        # effective_window_size = window_size if 1>=window_size else i
    
        window = np.ones(effective_window_size)

        res = np.sum(sequence[i-effective_window_size:i] * window)/effective_window_size
        # res = np.sum(sequence[i-effective_window_size:i] * window)/effective_window_size

        occurence_frequency.append(res)

    return occurence_frequency