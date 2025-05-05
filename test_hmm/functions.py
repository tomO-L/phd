#######################
### Import packages ###
#######################

import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from hmmlearn import hmm


########################
### Define functions ###
########################

# Data manipulation #
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

def load_data(folder_path_mouse_to_analyse,session_index):

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

def extract_runs_sequence(path_to_data_folder, mouse, session_index):

    ### Parameters ###

    mouse_folder_path = os.path.join(path_to_data_folder, mouse)
    epoch_types = ['run_around_tower', 'run_between_towers', 'run_toward_tower', 'exploratory_run']

    ### Extract epochs ###

    data = load_data(mouse_folder_path, session_index)

    ordered_epochs, ordered_epochs_frames = order_epochs(data['all_epochs'])

    ordered_runs_types = []
    ordered_runs_frames = []

    for i in range(len(ordered_epochs)):

        ordered_runs_types.append(ordered_epochs[i][0])
        ordered_runs_frames.append(ordered_epochs_frames[i])

    while 'immobility' in ordered_runs_types:

        first_occurence = ordered_runs_types.index('immobility')
        ordered_runs_types.remove('immobility')
        ordered_runs_frames.pop(first_occurence)

    num_epoch = len(ordered_runs_types)

    ordered_runs_types_number = np.nan * np.ones(num_epoch)

    for i, e_type in enumerate(epoch_types):
        ordered_runs_types_number = np.where(np.array(ordered_runs_types)==e_type, i, ordered_runs_types_number)

    return ordered_runs_types_number, ordered_runs_frames

































def plot_runs_distribution(ax, ordered_runs_types_number):

    epoch_types = ['run_around_tower', 'run_between_towers', 'run_toward_tower', 'exploratory_run']

    gen_epoch_types_ditribution = [np.count_nonzero(ordered_runs_types_number==i) for i in range(4)]
                                
    ax.bar(epoch_types, gen_epoch_types_ditribution)



def plot_runs_sequence(ax, ordered_runs_types_number, ordered_runs_frames=[]):

    epoch_types = ['run_around_tower', 'run_between_towers', 'run_toward_tower', 'exploratory_run']
    num_runs = len(ordered_runs_types_number)

    for i in range(len(epoch_types)):


        if len(ordered_runs_frames)==0:

            ordered_runs_type_number = np.where(np.array(ordered_runs_types_number)==i, np.arange(num_runs)-0.25, np.nan)
            x_barh = np.transpose([ordered_runs_type_number,0.5*np.ones(num_runs)])
        
        else:

            ordered_runs_frames = np.array(ordered_runs_frames)
            ordered_runs_frame = np.where(np.array(ordered_runs_types_number)==i, ordered_runs_frames[:,0], np.nan)
            ordered_runs_width = np.where(np.array(ordered_runs_types_number)==i, ordered_runs_frames[:,1] - ordered_runs_frames[:,0], np.nan)

            x_barh = np.transpose([ordered_runs_frame,ordered_runs_width])

        y_barh = [i-0.25,0.5]

        ax.broken_barh(x_barh, y_barh)

    ax.set_yticks(np.arange(len(epoch_types)), epoch_types)

















def infer_best_model(x_train, x_validate, lengths, n_to_test, seed=13):
    # check optimal score

    best_score = best_model = None
    n_fits = 200
    np.random.seed(seed)

    for n in n_to_test:
        for idx in range(n_fits):
            model = hmm.CategoricalHMM(
                n_components=n, random_state=idx,
                init_params='ste', algorithm='viterbi', n_features=4)  # don't init transition, set it below
            # we need to initialize with random transition matrix probabilities
            # because the default is an even likelihood transition
            # we know transitions are rare (otherwise the casino would get caught!)
            # so let's have an Dirichlet random prior with an alpha value of
            # (0.1, 0.9) to enforce our assumption transitions happen roughly 10%
            # of the time

            # transmat = []
            # for _ in range(n):

            #     row = np.random.uniform(size=n)
            #     row = row/np.sum(row)
                
            #     transmat.append(row)

            # transmat = np.array(transmat)
            # model.transmat_ = transmat
            
            model.fit(x_train, lengths)
            score = model.score(x_validate)
            # print(f'Model {n} components #{idx}\tScore: {score}')
            if best_score is None or score > best_score:
                best_model = model
                best_score = score

    return best_model, best_score
