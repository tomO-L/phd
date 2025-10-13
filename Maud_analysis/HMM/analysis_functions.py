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

def compute_occurence_frequency_v2(sequence,window_size=50):

    occurence_frequency = []

    for i in range(len(sequence)):

        effective_window_size = window_size if i>=window_size else i
        # effective_window_size = window_size if 1>=window_size else i
    
        res = np.sum(sequence[i-effective_window_size:i])/effective_window_size
        # res = np.sum(sequence[i-effective_window_size:i] * window)/effective_window_size

        occurence_frequency.append(res)

    return occurence_frequency

def count_tats(tats):

    total_tats = len(tats)
    cw_tats = 0
    rewarded_tats = 0

    for tat in tats:

        if tat[3]['direction']=='CW':

            cw_tats += 1

        if tat[3]['Rewarded']:

            rewarded_tats += 1

    return total_tats, cw_tats, rewarded_tats

def compute_rewards_persession(path_to_data_folder, mice_to_analyse):

    # Initialize dictionaries to store the various metrics for each mouse
    rewarded_tats_persession = []
    unrewarded_tats_persession = []

    # Iterate through each mouse to process its data
    for mouse in mice_to_analyse:
        folder_path_mouse_to_process = os.path.join(path_to_data_folder, mouse)

        # Get the list of sessions for the current mouse
        sessions_to_process = sorted([name for name in os.listdir(folder_path_mouse_to_process)
                                    if os.path.isdir(os.path.join(folder_path_mouse_to_process, name))
                                    and name.startswith('MOU')])

        # limit the analysis to the subseet of session we want to analyse
        # sessions_to_process = sessions_to_process[first_and_last_session_indexes[0]:first_and_last_session_indexes[1]]

        # nb_sessions = len(sessions_to_process)
        # print(f'Mouse {mouse}. There is/are {nb_sessions} sessions:')
        # print(sessions_to_process, '\n')


        # Process each session for the current mouse
        for session_index, session_to_process in enumerate(sessions_to_process):

            
            # Define the pickle file path for the session
            output_pickle_filename = f"{session_to_process}_basic_processing_output.pickle"
            output_pickle_filepath = os.path.join(folder_path_mouse_to_process, session_to_process, output_pickle_filename)

            # Check if the pickle file exists
            if not os.path.exists(output_pickle_filepath):
                print(f'Pickle file does not exist for session {session_to_process}, skipping .....')
                # Append session Nan data to the respective dictionaries
                rewarded_tats_persession.append([session_index + 1, np.nan])
                unrewarded_tats_persession.append([session_index + 1, np.nan])
                continue  # Skip to the next session if the pickle file does not exist

            # Load the pickle file
            with open(output_pickle_filepath, 'rb') as file:
                session_data = pickle.load(file)
            
            # Extract run around tower results from the session data
            
            epochs = session_data['all_epochs']
            tats = epochs['run_around_tower']
            total_tats, cw_tats, rewarded_tats = count_tats(tats)

            # Append session data to the respective dictionaries
            rewarded_tats_persession.append([session_index + 1, rewarded_tats])
            unrewarded_tats_persession.append([session_index + 1, total_tats-rewarded_tats])

    return rewarded_tats_persession, unrewarded_tats_persession
