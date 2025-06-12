#######################
### Import packages ###
#######################

import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from hmmlearn import hmm, vhmm
from tqdm import tqdm

########################
### Define functions ###
########################

# Data  #
def order_epochs(all_epochs):

    """
    Sort epochs in chronological order, omitting immobility epochs.

    Arguments:
        all_epoch (dict): dictionnary containing all the epochs, sorted by type of epoch. Each key is a different type of epoch.

    Returns:
        list: ordered_all_runs containing all the epochs omitting immobility epochs, sorted in chronological way.
        list: ordered_all_runs_frames containing all the epoch's frame intervals, omitting immobility epochs, sorted in chronological way.

    """

    # Initialize empty lists to store ordered runs and their first frames
    ordered_all_epochs = []
    ordered_all_epochs_frames = []

    # Loop through each key in the all_epochs dictionary
    for k in all_epochs.keys():
        
        # Loop through each run in the current key's list
        for i in range(len(all_epochs[k])):
            
            # Treats immobility differently because of a structure difference of the epoch variables
            if k != 'immobility':
        
                # Add the current run to ordered_all_runs
                ordered_all_epochs.append([k] + all_epochs[k][i])
                # Add the first frame of the current run to ordered_all_runs_frames
                ordered_all_epochs_frames.append(all_epochs[k][i][0])

            else: 
            
                # TODO: deal with this hellish format

                start_frame = all_epochs[k][i][0]
                end_frame = all_epochs[k][i][1]

                reformated_epoch = all_epochs[k][i].copy()
                reformated_epoch.remove(reformated_epoch[0])
                reformated_epoch[0] = [start_frame,end_frame]

                # Add the current run to ordered_all_runs                
                ordered_all_epochs.append([k] + reformated_epoch)
                # Add the first frame of the current run to ordered_all_runs_frames
                ordered_all_epochs_frames.append(reformated_epoch[0])

    # Sort the frames list based on the first element of each frame
    ordered_all_epochs_frames = sorted(ordered_all_epochs_frames, key=lambda x: x[1])
    # Sort the runs list based on the first element of each run
    ordered_all_epochs = sorted(ordered_all_epochs, key=lambda x: x[1])

    # Return the ordered lists of runs and their first frames
    return ordered_all_epochs, ordered_all_epochs_frames

def identify_action(epoch):

    """
    Identify action type 
    """

    action_types = ['run_around_tower_CW', 'run_around_tower_CCW', 'run_toward_tower', 'exploratory_run']

    if epoch[0]=='run_around_tower':

        direction = epoch[4]['direction']
        action_name = f'{epoch[0]}_{direction}'

    elif epoch[0]=='run_between_towers' or epoch[0]=='run_toward_tower':

        action_name = 'run_toward_tower'

    else:

        action_name = epoch[0]

    action = action_types.index(action_name)

    return action, action_types

# def identify_action(epoch):

#     """
#     Identify action type 
#     """

#     action_types = ['run_around_tower_CW', 'run_around_tower_CCW', 'run_between_towers', 'run_toward_tower', 'exploratory_run']

#     if epoch[0]!='run_around_tower':

#         action_name = epoch[0]

#     else:

#         direction = epoch[4]['direction']
#         action_name = f'{epoch[0]}_{direction}'

#     action = action_types.index(action_name)

#     return action, action_types

# def identify_action(epoch):

#     """
#     Identify action type 
#     """

#     action_types = ['run_around_tower', 'run_between_towers', 'run_toward_tower', 'exploratory_run']

#     action_name = epoch[0]


#     action = action_types.index(action_name)

#     return action, action_types


def load_pickle_data(folder_path_mouse_to_analyse,session_index):

    """
    Load pickle data file

    Arguments:
        folder_path_mouse_to_analyse (str): path to mouse folder
        session_index (int): index of the session from which to load the pickle data file
    
    Returns:
        (list) data from pickle file 

    """

    # Get all session folders that start with 'MOU' and sort them
    sessions_to_analyse = sorted([name for name in os.listdir(folder_path_mouse_to_analyse)
                                  if os.path.isdir(os.path.join(folder_path_mouse_to_analyse, name))
                                  and name.startswith('MOU')])

    session_to_analyse = sessions_to_analyse[session_index]

    # Define the output pickle filename and its full path
    output_pickle_filename = f"{session_to_analyse}_basic_processing_output.pickle"        
    output_pickle_filepath = os.path.join(folder_path_mouse_to_analyse, session_to_analyse, output_pickle_filename)

    # Open and load the session data from the pickle file
    with open(output_pickle_filepath, 'rb') as file:
        session_data = pickle.load(file)

    return session_data

def extract_actions_sequence(path_to_data_folder, mouse, session_index):

    ### Parameters ###

    mouse_folder_path = os.path.join(path_to_data_folder, mouse)

    ### Extract epochs ###

    data = load_pickle_data(mouse_folder_path, session_index)

    ordered_epochs, ordered_epochs_frames = order_epochs(data['all_epochs'])

    ordered_runs_types_number = []
    ordered_runs_frames = []

    for i in range(len(ordered_epochs)):

        epoch_name = ordered_epochs[i][0]

        if epoch_name=='immobility':

            continue

        action, action_types = identify_action(ordered_epochs[i])

        ordered_runs_types_number.append(action)
        ordered_runs_frames.append(ordered_epochs_frames[i])
    
    return np.array(ordered_runs_types_number), np.array(ordered_runs_frames), action_types
