
import os
import pickle
import numpy as np
import pandas as pd
import ast
import matplotlib.path as mpath
from scipy.stats import wilcoxon

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




















def get_trapeze_and_tower_data(folder_path_mouse_to_process, session_to_process):

    """
    Function to extract trapeze width and tower coordinates from a session parameter CSV file.
    
    Parameters:
        folder_path_mouse_to_process (str): The folder path where the mouse data is stored.
        session_to_process (str): The specific session to process.

    Returns:
        trapeze_width (int or float): The width of the trapeze.
        towers_coordinates (dict): The coordinates of the towers.
    """
    # Load the session parameters CSV file
    param_file_path = os.path.join(folder_path_mouse_to_process, session_to_process, f"{session_to_process}_sessionparam.csv")
    param_df = pd.read_csv(param_file_path)

    # Check if the towers coordinates exist in the CSV file
    if "SE_coords" in param_df.columns:
        towers_coordinates = {
            "NW": param_df["NW_coords"].values[0],
            "NE": param_df["NE_coords"].values[0],
            "SW": param_df["SW_coords"].values[0],
            "SE": param_df["SE_coords"].values[0]
        }
        
        # Convert string representations of lists into actual lists
        towers_coordinates = {key: ast.literal_eval(value) for key, value in towers_coordinates.items()}

        #print('Coordinates from parameter file:')
    else:
        # Default tower coordinates
        towers_coordinates = {
            "NW": [[104, 125], [173, 125], [173, 201], [104, 201]],
            "NE": [[330, 120], [400, 120], [400, 200], [330, 200]],
            "SW": [[109, 351], [181, 351], [181, 410], [109, 410]],
            "SE": [[330, 350], [400, 350], [400, 410], [330, 410]]
        }
        towers_coordinates = {
            "NW": [[114, 125], [183, 125], [183, 201], [114, 201]],
            "NE": [[330, 120], [400, 120], [400, 200], [330, 200]],
            "SW": [[119, 351], [191, 351], [191, 410], [119, 410]],
            "SE": [[327, 350], [397, 350], [397, 410], [327, 410]]
        }
        #small modification due to Aurelien's setting
    #print(towers_coordinates)

    # Check if the trapeze size exists in the CSV file
    if "TrapezeSize" in param_df.columns:
        trapeze_width = param_df["TrapezeSize"].values[0]
        #print('Trapeze width from parameter file:')
    else:
        trapeze_width = 50  # Default trapeze width
        #print('Default trapeze width')
    #print(trapeze_width)

    return trapeze_width, towers_coordinates

def generate_trapeze_and_tower_coordinates(towers_coordinates, trapeze_width):
    """
    Generates the coordinates of trapezes surrounding towers and converts all coordinates from pixels to centimeters.
    
    Parameters:
    towers_coordinates (dict): Dictionary containing the pixel coordinates of the 4 towers.
    trapeze_width (int): The width of the trapeze in pixels.
    
    
    Returns:
    tuple: 
        - all_trapezes_coordinates_cm (dict): Coordinates of the trapezes in cm.
        - towers_coordinates_cm (dict): Coordinates of the towers in cm.
    """
    
    #video_dimension_pixels (tuple): The resolution of the video in pixels 
    #arena_width_cm (float): The width of the arena in centimeters 
    #arena_width_pixels (int): The width of the arena in pixels 
    video_dimension_pixels=(512, 512)
    arena_width_cm=84
    arena_width_pixels=453



    # Conversion factor to go from pixel to cm
    conversion_factor = arena_width_cm / arena_width_pixels
    
    # Function to convert pixel coordinates to cm
    def convert_pix_to_cm(coordinate):
        return [round(coordinate[0] * conversion_factor, 2), round(coordinate[1] * conversion_factor, 2)]
    
    # Transform the coordinates to have the origin at the lower left (for plotting)
    max_y = video_dimension_pixels[1]
    transformed_towers_coordinates = {
        label: [[x, max_y - y] for x, y in vertices]
        for label, vertices in towers_coordinates.items()
    }
    
    # Function to generate trapeze coordinates surrounding a tower
    def trapeze_coordinates_from_tower(tower_coordinates, trapeze_width):
        trapeze_N = [
            tower_coordinates[0], tower_coordinates[1],
            [tower_coordinates[1][0] + trapeze_width, tower_coordinates[1][1] + trapeze_width],
            [tower_coordinates[0][0] - trapeze_width, tower_coordinates[0][1] + trapeze_width]
        ]
        trapeze_E = [
            tower_coordinates[1], tower_coordinates[2],
            [tower_coordinates[2][0] + trapeze_width, tower_coordinates[2][1] - trapeze_width],
            [tower_coordinates[1][0] + trapeze_width, tower_coordinates[1][1] + trapeze_width]
        ]
        trapeze_S = [
            tower_coordinates[2], tower_coordinates[3],
            [tower_coordinates[3][0] - trapeze_width, tower_coordinates[3][1] - trapeze_width],
            [tower_coordinates[2][0] + trapeze_width, tower_coordinates[2][1] - trapeze_width]
        ]
        trapeze_W = [
            tower_coordinates[3], tower_coordinates[0],
            [tower_coordinates[0][0] - trapeze_width, tower_coordinates[0][1] + trapeze_width],
            [tower_coordinates[3][0] - trapeze_width, tower_coordinates[3][1] - trapeze_width]
        ]
        return trapeze_N, trapeze_E, trapeze_S, trapeze_W
    
    # Initialize dictionaries to store trapeze and tower coordinates in cm
    all_trapezes_coordinates = {key: {} for key in towers_coordinates}
    all_trapezes_coordinates_cm = {}
    
    # Generate trapeze coordinates for each tower
    for tower_name, tower_coordinates in transformed_towers_coordinates.items():
        all_trapezes_coordinates[tower_name]["N"], \
        all_trapezes_coordinates[tower_name]["E"], \
        all_trapezes_coordinates[tower_name]["S"], \
        all_trapezes_coordinates[tower_name]["W"] = trapeze_coordinates_from_tower(tower_coordinates, trapeze_width)

    # Convert all trapeze coordinates from pixel to cm
    for tower, trapezes in all_trapezes_coordinates.items():
        all_trapezes_coordinates_cm[tower] = {
            trapeze: [convert_pix_to_cm(coord) for coord in coords]
            for trapeze, coords in trapezes.items()
        }
    
    # Convert tower coordinates from pixel to cm
    towers_coordinates_cm = {
        key: [convert_pix_to_cm(coord) for coord in transformed_towers_coordinates[key]]
    for key in transformed_towers_coordinates}

    return all_trapezes_coordinates_cm, towers_coordinates_cm



































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



def one_sample_wilcoxon_test(sample, med0):

    w, p = wilcoxon(np.array(sample) - med0)

    return p

def plot_runs_sequence(ax, ordered_runs_types_number, cmaps=['black', 'black', 'black', 'black']):

    epoch_types = ['run_around_tower', 'run_between_towers', 'run_toward_tower', 'exploratory_run']
    num_runs = len(ordered_runs_types_number)

    for i in range(len(epoch_types)):

        ordered_runs_type_number = np.where(np.array(ordered_runs_types_number)==i, np.arange(num_runs)-0.1, np.nan)
        x = ordered_runs_type_number

        y = i * np.ones(len(x))

        ax.scatter(x, y, c=cmaps, marker='|')

    ax.set_yticks(np.arange(len(epoch_types)), epoch_types)


def plot_learning_curve(values_persessions_permouse, mice_to_analyse, ax, wilcoxon_test_h0=np.nan):

    all_mice_values_persessions = []

    for mouse in mice_to_analyse:

        values_persessions = values_persessions_permouse[mouse]

        values_persessions = np.transpose(values_persessions)
        all_mice_values_persessions.append(values_persessions[1])

        # Plot individual mice
        ax.plot(values_persessions[0],values_persessions[1], alpha=0.3)

    # Plot median and quartiles
    median_values = np.nanmedian(all_mice_values_persessions, axis=0)
    upper_quartile_values = np.nanpercentile(all_mice_values_persessions, 75, axis=0)
    lower_quartile_values = np.nanpercentile(all_mice_values_persessions, 25, axis=0)

    # Perform one sample Wilcoxon test

    if not(np.isnan(wilcoxon_test_h0)):

        for i in range(len(values_persessions[0])):

            p = one_sample_wilcoxon_test(np.transpose(all_mice_values_persessions)[i], wilcoxon_test_h0)

            # print(p)
        
            if p<=0.05:

                ax.scatter(i+1,median_values[i],color='#1f77b4', edgecolor='black', marker='*',s=20, linewidths=0.5, zorder=1000)

    ax.errorbar(values_persessions[0],median_values, yerr=[median_values-lower_quartile_values, upper_quartile_values-median_values], color='black')
    
    ax.set_xticks(values_persessions[0])









