
import os
import pickle
import numpy as np
import pandas as pd

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
    Determines the rewarded direction for the last session of a given mouse.
    In the protocol this notebook is used for, the rewarded direction of the 
    last session is the same as in all the sessions where rewarding is allowed. 
    
    Arguments:
        folder_path_mouse_to_process (str): Path to the folder containing mouse sessions folders.
        session_index (int): Index of the session that will be used to define the rewarded direction
        start_session_index (int, optional): index of the first session in the list of sessions to analyse. 
                                   If different from 0, the session index should refer to 
                                   the index of the session in the sub-list of session to process, not the total list 

    Returns:
        str: 'CW' (Clockwise) if the rewarded direction is 270 degrees,
             'CCW' (Counterclockwise) if the rewarded direction is 90 degrees,
             numpy.nan if reward delivery is not allowed 
             None if an error occurs.
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
            rewarded_direction = 'both'  # Both directions

        else:
            print('ERROR: Unexpected unique rewarded direction value:', rewarded_direction_degrees)
            return None  # Explicitly return None to indicate failure
    
    else:

        # Rewarded direction is set to X if reward delivery os not allowed
        rewarded_direction = 'X'

    return rewarded_direction


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



























def order_runs(all_epochs):

    ordered_all_runs = []
    ordered_all_runs_frames = []

    for k in all_epochs.keys():

        if k != 'immobility':

            for i in range(len(all_epochs[k])):

                ordered_all_runs.append(all_epochs[k][i])
                ordered_all_runs_frames.append(all_epochs[k][i][0])

    ordered_all_runs_frames = sorted(ordered_all_runs_frames, key=lambda x: x[0])
    
    ordered_all_runs = sorted(ordered_all_runs,key=lambda x: x[0])

    return ordered_all_runs, ordered_all_runs_frames

def find_visits(all_epochs):

    visits = []

    n = -1

    runs_around_tower = all_epochs['run_around_tower']

    ordered_all_runs, ordered_all_runs_frames = order_runs(all_epochs)


    for run_around_tower in runs_around_tower:

        is_good_turn = run_around_tower[3]['Rewarded']
        max_rewards = run_around_tower[3]['max_rewards']
        # patch = run_around_tower[1][0]

        ordered_idx = ordered_all_runs_frames.index(run_around_tower[0])

        departure, arrival = [ordered_all_runs[ordered_idx-1][1][0],ordered_all_runs[ordered_idx-1][2][0]] if ordered_idx !=0 else ['','']

        is_previous_run_not_a_turn = (departure != arrival) or len(visits)==0

        if is_previous_run_not_a_turn:
        # if patch != previous_turn_patch:

            n = n + 1
            
            visits.append({})

            visits[n]['turns'] = 1
            visits[n]['rewarded_turns'] = int(is_good_turn)
            visits[n]['max_reward'] = max_rewards
            visits[n]['patch'] = run_around_tower[1][0] #patch
            visits[n]['visit_time'] = run_around_tower[4]['epoch_time']


        else:

            visits[n]['rewarded_turns'] += int(is_good_turn)
            visits[n]['turns'] += 1

        #previous_turn_patch = patch

    #print(visit)

    return visits

def compute_turns_per_rewarded_visit(folder_path_mouse_to_analyse, session_index):

    # Get all session folders that start with 'MOU' and sort them
    sessions_to_analyse = sorted([name for name in os.listdir(folder_path_mouse_to_analyse)
                                  if os.path.isdir(os.path.join(folder_path_mouse_to_analyse, name))
                                  and name.startswith('MOU')])

    session_to_analyse = sessions_to_analyse[session_index]

    output_pickle_filename = f"{session_to_analyse}_basic_processing_output.pickle"
    output_pickle_filepath = os.path.join(folder_path_mouse_to_analyse, session_to_analyse, output_pickle_filename)

    rewarded_direction = finding_mouse_rewarded_direction(folder_path_mouse_to_analyse, session_index)
            
    # Load the pickle file
    with open(output_pickle_filepath, 'rb') as file:
        session_data = pickle.load(file)

    #runs_around_towers = session_data['all_epochs']['run_around_tower']
        
    visit = find_visits(session_data['all_epochs'])
    
    # print(len(visit), ' number of visits')

    turns_per_visit = []
    rewarded_turns_per_visit = []
    visits_time = []
    max_rewards = []

    for i in range(len(visit)):

        if rewarded_direction=='X':

            nb_of_turns = visit[i]['turns']
            nb_of_rewarded_turns = np.nan # visit[i]['rewarded_turns']
            visit_time = visit[i]['visit_time']
            max_reward = np.nan # visit[i]['max_reward']


        else:
            
            nb_of_turns = visit[i]['turns']
            nb_of_rewarded_turns = visit[i]['rewarded_turns']
            visit_time = visit[i]['visit_time']
            max_reward = visit[i]['max_reward']

        turns_per_visit.append(nb_of_turns)
        rewarded_turns_per_visit.append(nb_of_rewarded_turns)
        visits_time.append(visit_time)
        max_rewards.append(max_reward)


    return [np.array(turns_per_visit),np.array(rewarded_turns_per_visit),np.array(visits_time),np.array(max_rewards)]

def extract_runs_sequence(path_to_data_folder, mouse, session_index):

    ### Parameters ###

    mouse_folder_path = os.path.join(path_to_data_folder, mouse)
    epoch_types = ['run_around_tower', 'run_between_towers', 'run_toward_tower', 'exploratory_run']

    ### Extract epochs ###

    data = load_pickle_data(mouse_folder_path, session_index)

    ordered_epochs, ordered_epochs_frames = order_epochs(data['all_epochs'])

    ordered_runs = []
    ordered_runs_types = []
    ordered_runs_frames = []

    for i in range(len(ordered_epochs)):

        ordered_runs.append(ordered_epochs[i])
        ordered_runs_types.append(ordered_epochs[i][0])
        ordered_runs_frames.append(ordered_epochs_frames[i])

    while 'immobility' in ordered_runs_types:

        first_occurence = ordered_runs_types.index('immobility')
        ordered_runs_types.remove('immobility')
        ordered_runs.pop(first_occurence)
        ordered_runs_frames.pop(first_occurence)

    num_epoch = len(ordered_runs_types)

    ordered_runs_types_number = np.nan * np.ones(num_epoch)

    for i, e_type in enumerate(epoch_types):
        ordered_runs_types_number = np.where(np.array(ordered_runs_types)==e_type, i, ordered_runs_types_number)

    return ordered_runs, ordered_runs_types_number, ordered_runs_frames






























# def plot_runs_sequence(ax, ordered_runs_types_number, cmaps=['black', 'black', 'black', 'black'], ordered_runs_frames=[]):

#     epoch_types = ['run_around_tower', 'run_between_towers', 'run_toward_tower', 'exploratory_run']
#     num_runs = len(ordered_runs_types_number)

#     for i in range(len(epoch_types)):

#         if len(ordered_runs_frames)==0:

#             ordered_runs_type_number = np.where(np.array(ordered_runs_types_number)==i, np.arange(num_runs)-0.01, np.nan)
#             x_barh = np.transpose([ordered_runs_type_number,0.02*np.ones(num_runs)])
        
#         else:

#             ordered_runs_frames = np.array(ordered_runs_frames)
#             ordered_runs_frame = np.where(np.array(ordered_runs_types_number)==i, ordered_runs_frames[:,0], np.nan)
#             ordered_runs_width = np.where(np.array(ordered_runs_types_number)==i, ordered_runs_frames[:,1] - ordered_runs_frames[:,0], np.nan)

#             x_barh = np.transpose([ordered_runs_frame,ordered_runs_width])

#         y_barh = [i-0.25,0.5]

#         ax.broken_barh(x_barh, y_barh, color=cmaps[i])

#     ax.set_yticks(np.arange(len(epoch_types)), epoch_types)



def plot_runs_sequence(ax, ordered_runs_types_number, cmaps=['black', 'black', 'black', 'black']):

    epoch_types = ['run_around_tower', 'run_between_towers', 'run_toward_tower', 'exploratory_run']
    num_runs = len(ordered_runs_types_number)

    for i in range(len(epoch_types)):

        ordered_runs_type_number = np.where(np.array(ordered_runs_types_number)==i, np.arange(num_runs)-0.1, np.nan)
        x = ordered_runs_type_number

        y = i * np.ones(len(x))

        ax.scatter(x, y, c=cmaps, marker='|')

    ax.set_yticks(np.arange(len(epoch_types)), epoch_types)