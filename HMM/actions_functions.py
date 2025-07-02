#######################
### Import packages ###
#######################

import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import pandas as pd
from hmmlearn import hmm, vhmm
from tqdm import tqdm

########################
### Define functions ###
########################

# Data #
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

# def identify_action(epoch):

#     """
#     Identify action type (1)
#     """

#     action_types = ['run_around_tower', 'run_between_towers', 'run_toward_tower', 'exploratory_run']

#     action_name = epoch[0]


#     action = action_types.index(action_name)

#     return action, action_types

def identify_action(epoch):

    """
    Identify action type (2)
    """

    action_types = ['run_around_tower_CW', 'run_around_tower_CCW', 'run_between_towers', 'run_toward_tower', 'exploratory_run']

    if epoch[0]!='run_around_tower':

        action_name = epoch[0]

    else:

        direction = epoch[4]['direction']
        action_name = f'{epoch[0]}_{direction}'

    action = action_types.index(action_name)

    return action, action_types


# def identify_action(epoch):

#     """
#     Identify action type (3)
#     """

#     action_types = ['run_around_tower_CW', 'run_around_tower_CCW', 'run_toward_tower', 'exploratory_run']

#     if epoch[0]=='run_around_tower':

#         direction = epoch[4]['direction']
#         action_name = f'{epoch[0]}_{direction}'

#     elif epoch[0]=='run_between_towers' or epoch[0]=='run_toward_tower':

#         action_name = 'run_toward_tower'

#     else:

#         action_name = epoch[0]

#     action = action_types.index(action_name)

#     return action, action_types

def load_csv_data(mouseFolder_Path, session):

    trajectory_df, turns_df, param_df = None, None, None  # Initialize as None


    try:
        # Gets the parameters of the session
        param_df = pd.read_csv(mouseFolder_Path + os.sep + session + os.sep + session + "_sessionparam.csv")
    except FileNotFoundError:
        print("File sessionparam not found")

    try:
        #Gets the positional informations and filter the dataframe to keep only the relevant informations
        csvCentroid_fullpath = mouseFolder_Path + os.sep + session + os.sep + session + '_centroidTXY.csv'
        trajectory_df = pd.read_csv(csvCentroid_fullpath) #Transforms CSV file into panda dataframe
        trajectory_df = trajectory_df.dropna() #Deletes lines with one or more NA
        trajectory_df = trajectory_df.loc[trajectory_df['time'] > 15] #During the first seconds of the video, as the background substraction is still building up, 
        #                                           #the tracking is innacruate so we don't analyze postions during the first 15 seconds
        trajectory_df = trajectory_df[trajectory_df['xposition'].between(1, 500) & trajectory_df['yposition'].between(1, 500)] #The pixel values between 1 and 500 are kept)
    except FileNotFoundError:
        print("File centroidTXY not found")

    try:
        # Get the information on the turns
        csvTurnsinfo_fullpath = mouseFolder_Path + os.sep + session + os.sep + session + '_turnsinfo.csv'  # get the information on the turns in the dataframe turns_df
        turns_df = pd.read_csv(csvTurnsinfo_fullpath)  # Transforms CSV file into panda dataframe
        for i in range(turns_df.index.values[-1]):  # if there is a missing value for ongoingRewardedObject, replace it with either SW or SE, as long as it's not the one where the mouse is
            if type(turns_df['ongoingRewardedObject'][i]) == float:
                turns_df.iat[i, 8] = str([turns_df.iat[i, 4]])
        turns_df = turns_df.loc[turns_df['time'] > 15]  # same as above #TODO someone you shoud spend some time on the aquisition code to have a pre-loaded background and not loose the beginning
    except FileNotFoundError:
        print("File turnsinfo not found")

    return trajectory_df, turns_df, param_df


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


def finding_mouse_rewarded_direction(folder_path_mouse_to_process, session_index):
    
    """
    Determines the rewarded direction for the session corresponding to the input index of a given mouse. 
    This index is relative to the session position in the series of analysed sessions.
    This input index can have an offset. This is usefull if the sessions series analysed does not start with the first session. 
    
    Arguments:
        folder_path_mouse_to_process (str): Path to the folder containing mouse sessions folders.
        session_index (int): Index of the session that will be used to define the rewarded direction

    Returns:
        str: 'CW' (Clockwise) if the rewarded direction is 270 degrees,
             'CCW' (Counterclockwise) if the rewarded direction is 90 degrees,
             numpy.nan if reward delivery is not allowed 
             None if an error occurs or if both directions are rewarded.
    """
    
    # Get all session folders that start with 'MOU' and sort them
    sessions_to_process = sorted([name for name in os.listdir(folder_path_mouse_to_process)
                                  if os.path.isdir(os.path.join(folder_path_mouse_to_process, name))
                                  and name.startswith('MOU')])

    # Load data from the last session
    session_traj_df, session_turns_df, session_param_df = load_csv_data(folder_path_mouse_to_process, sessions_to_process[session_index])

    # Extract rewarded direction in degrees
    rewarded_direction_degrees = session_param_df["potentialRewardedDirections"][0]

    # Check if reward delivery is allowed
    if session_param_df["allowRewardDelivery"][0]:

        # Determine the rewarded direction based on the extracted value
        if rewarded_direction_degrees == '[270]':
            rewarded_direction = 'CW'  # Clockwise

        elif rewarded_direction_degrees == '[90]':
            rewarded_direction = 'CCW'  # Counterclockwise
        
        elif rewarded_direction_degrees == '[90, 270]':
            rewarded_direction = 'both'  # Clockwise and Counterclockwise

        # Returns None if the rewarded direction entry in session_param_df is not recognized
        else:
            print('ERROR: Unexpected rewarded direction value:', rewarded_direction_degrees)
            return None  # Explicitly return None to indicate failure
    
    else:

        # Rewarded direction is set to X if reward delivery os not allowed
        rewarded_direction = 'X'

    return rewarded_direction

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
