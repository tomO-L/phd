#######################
### Import packages ###
#######################

import matplotlib.pyplot as plt
import sys
import time
from IPython.core import ultratb
import dill
import numpy as np
from tqdm import tqdm
import scipy.optimize as opt
from functions_synthetic_data_generation import *
sys.excepthook = ultratb.FormattedTB(call_pdb=False)

# plt.style.use('paper.mplstyle')

# Time counter
start_time = time.time()

#################
### Functions ###
#################


def compute_simulations_average(args_dict, n_simulations=300):

    synthetic_proba_list = []

    for _ in tqdm(range(n_simulations), leave=False):
    
        ddm_result = run_simulation(args_dict,)

        synthetic_proba_list.append(ddm_result['p_cw'])

    average_trajectory = np.mean(synthetic_proba_list,axis=0)

    return average_trajectory

def compute_simulations_average_random_init(args_dict, p_cw_init_range, n_simulations=300):

    synthetic_proba_list = []

    for _ in tqdm(range(n_simulations), leave=False):

        temp_args_dict = copy.deepcopy(args_dict)
        temp_args_dict['p_cw_init'] = np.random.choice(p_cw_init_range)

        ddm_result = run_simulation(temp_args_dict)

        synthetic_proba_list.append(ddm_result['p_cw'])

    average_trajectory = np.mean(synthetic_proba_list,axis=0)

    return average_trajectory

def compute_simulations_mean_square_displacement_switch(args_dict, n_simulations=300):

    msd_list = []

    p_cw_reward = args_dict['p_cw_reward']
    p_ccw_reward = args_dict['p_ccw_reward']


    for i in tqdm(range(n_simulations), leave=False):
    
        temp_args_dict = copy.deepcopy(args_dict)

        if i>=int(n_simulations/2):

            temp_args_dict['p_cw_reward'], temp_args_dict['p_ccw_reward'] = p_ccw_reward, p_cw_reward

        ddm_result = run_simulation(temp_args_dict)

        starting_proba = ddm_result['parameters']['p_cw_init']

        mean_square_displacement = (ddm_result['p_cw'] - starting_proba)**2

        msd_list.append(mean_square_displacement)

    mean_square_displacement_sequence = np.mean(msd_list, axis=0)

    return mean_square_displacement_sequence

def compute_simulations_mean_square_displacement(args_dict, n_simulations=300):

    msd_list = []

    for i in tqdm(range(n_simulations), leave=False):
    
        temp_args_dict = copy.deepcopy(args_dict)

        ddm_result = run_simulation(temp_args_dict)

        starting_proba = ddm_result['parameters']['p_cw_init']

        mean_square_displacement = (ddm_result['p_cw'] - starting_proba)**2

        msd_list.append(mean_square_displacement)

    mean_square_displacement_sequence = np.mean(msd_list, axis=0)

    return mean_square_displacement_sequence

def compute_mean_square_error(args):
    
    args_dict, n_simulations, reconstructed_average_trajectory = args

    steps_number = args_dict['steps_number']

    average_trajectory = compute_simulations_average(args_dict, n_simulations=n_simulations)

    mse = (np.sum((average_trajectory - reconstructed_average_trajectory))**2)/steps_number

    return mse

def generate_test_average_probability_sequences_identical_drifts_random_init(drift_range, args):

    args_dict, n_simulations, p_cw_init_range = args

    average_trajectory_list = []


    for drift in tqdm(drift_range):
    
        drift_matrix = np.array([[drift, -drift],
                                 [0    , 0     ]])

        args_dict['drift_matrix'] = drift_matrix

        average_trajectory = compute_simulations_average_random_init(args_dict, p_cw_init_range, n_simulations=n_simulations)
        average_trajectory_list.append(average_trajectory)

    return average_trajectory_list


def generate_test_average_probability_sequences_identical_drifts(drift_range, args):

    args_dict, n_simulations = args

    average_trajectory_list = []


    for drift in tqdm(drift_range):
    
        drift_matrix = np.array([[drift, -drift],
                                 [0    , 0     ]])

        args_dict['drift_matrix'] = drift_matrix

        average_trajectory = compute_simulations_average(args_dict, n_simulations=n_simulations)
        average_trajectory_list.append(average_trajectory)

    return average_trajectory_list
# def generate_test_msd_sequences_identical_drifts_switch(drift_range, args):

#     args_dict, n_simulations = args

#     msd_sequence_list = []

#     for drift in tqdm(drift_range):
    
#         drift_matrix = np.array([[drift, -drift],
#                                  [0    , 0     ]])

#         args_dict['drift_matrix'] = drift_matrix

#         msd_sequence = compute_simulations_mean_square_displacement_switch(args_dict, n_simulations=n_simulations)
#         msd_sequence_list.append(msd_sequence)

#     return msd_sequence_list

def generate_test_msd_sequences_identical_drifts(drift_range, args):

    args_dict, n_simulations = args

    msd_sequence_list = []

    for drift in tqdm(drift_range):
    
        drift_matrix = np.array([[drift, -drift],
                                 [0    , 0     ]])

        args_dict['drift_matrix'] = drift_matrix

        msd_sequence = compute_simulations_mean_square_displacement(args_dict, n_simulations=n_simulations)
        msd_sequence_list.append(msd_sequence)

    return msd_sequence_list

def compute_mean_square_error_v2(average_trajectory, reconstructed_average_trajectory):
    
    mse = (np.sum((average_trajectory - reconstructed_average_trajectory))**2)/len(average_trajectory)

    return mse

def compute_reconstructed_proba_sequence(choices_sequence, model):

    states_sequence = model.predict(np.int16(choices_sequence.reshape(-1,1)))
    # print(states_sequence)

    emissionprob = model.emissionprob_

    reconstructed_p_cw_sequence = []

    for s in states_sequence:

        reconstructed_p_cw_sequence.append(emissionprob[s][2]+emissionprob[s][3])

    return reconstructed_p_cw_sequence

def compute_reconstructed_average_proba_sequence(choices_sequences, model):

    sequences_number = len(choices_sequences)
    reconstructed_p_cw_sequences = []

    for i in range(sequences_number):
    
        choices_sequence = choices_sequences[i]
    
        reconstructed_p_cw_sequence = compute_reconstructed_proba_sequence(choices_sequence,model)

        reconstructed_p_cw_sequences.append(reconstructed_p_cw_sequence)

    reconstructed_average_p_cw = np.mean(reconstructed_p_cw_sequences,axis=0)

    return reconstructed_average_p_cw

def compute_reconstructed_mean_square_displacement_sequence(choices_sequences, model, starting_proba=None):

    if not(starting_proba):

        starting_proba = np.array(choices_sequences)

    sequences_number = len(choices_sequences)
    reconstructed_p_cw_sequences = []

    for i in range(sequences_number):
    
        choices_sequence = choices_sequences[i]
    
        reconstructed_p_cw_sequence = compute_reconstructed_proba_sequence(choices_sequence,model)

        reconstructed_p_cw_sequences.append(reconstructed_p_cw_sequence)

    mean_square_displacement_sequence = np.mean((np.arrray(reconstructed_p_cw_sequences) - starting_proba)**2)
    
    return mean_square_displacement_sequence



def compute_reconstructed_average_proba_sequences(n_simulations_list):

    reconstructed_average_proba_sequences = []

    for n_simulations in n_simulations_list:
        
        slice_size = int(n_simulations/4)

        with open(f'DDM/statistical_precision_analysis/simulations_batches/simulations_batch_{n_simulations}.pkl', 'rb') as file:
            synthetic_data = dill.load(file)

        test_data = [synth_data['choices'] for synth_data in synthetic_data][:2*slice_size]

        with open(f'DDM/statistical_precision_analysis/simulations_batches/best_model_score_{n_simulations}.pkl', 'rb') as file:
            model = dill.load(file)

        reconstructed_average_proba_sequences.append(compute_reconstructed_average_proba_sequence(test_data, model))

    return reconstructed_average_proba_sequences

def infer_probability_sequence(model, actions_sequences):

    states_sequences = []
    sequences_number = len(actions_sequences)

    for i in range(sequences_number):
        
        choices_sequence = actions_sequences[i]
        
        states_sequence = model.predict(np.int16(choices_sequence.reshape(-1,1)))
        states_sequences.append(states_sequence)

    emissionprob = model.emissionprob_


    reconstructed_p_a_sequences = []

    for i in range(len(states_sequences)):

        reconstructed_p_a_sequence = []

        for s in states_sequences[i]:

            reconstructed_p_a_sequence.append(emissionprob[s][1])

        reconstructed_p_a_sequences.append(reconstructed_p_a_sequence)

    return reconstructed_p_a_sequences

def count_initial_state_occurences(model,test_data):
    
    initial_state_list = []
    sequences_number = len(test_data)

    for i in range(sequences_number):
        
        choices_sequence = test_data[i]
        
        states_sequence = model.predict(np.int16(choices_sequence.reshape(-1,1)))
        initial_state_list.append(states_sequence[0])

    return initial_state_list

def compute_initial_state_list_distri(model,test_data):

    sequences_number = len(test_data)

    initial_state_list = count_initial_state_occurences(model, test_data)

    initial_state_list_distri = []

    for s in range(len(model.transmat_)):

        initial_state_list_distri.append(initial_state_list.count(s))

    initial_state_list_distri = np.array(initial_state_list_distri)/sequences_number

    return initial_state_list_distri

def plot_hmm(model,test_data):

    n_simulations = len(test_data)

    initial_state_list_distri = compute_initial_state_list_distri(model,test_data)

    transmat = model.transmat_
    emissionmat = model.emissionprob_

    fig=plt.figure(figsize=(5, 3), dpi=300, constrained_layout=False, facecolor='w')
    gs = fig.add_gridspec(1, 1)
    row = gs[0].subgridspec(1,3, width_ratios=(1,0.33,0.33))
    ax1 = plt.subplot(row[0,0])
    ax2 = plt.subplot(row[0,1])
    ax3 = plt.subplot(row[0,2])

    states_number = len(model.transmat_)

    ax1.imshow(transmat, vmin=0, vmax=1)
    ax1.set_xticks(np.arange(states_number))
    ax1.set_yticks(np.arange(states_number))

    ax1.set_title(f'Transition matrix\n{n_simulations} simulations', fontsize=7)
    ax1.set_xlabel('To state')
    ax1.set_ylabel('From state')

    ax2.imshow(emissionmat, vmin=0, vmax=1)
    ax2.set_xticks([0,1], labels=['CCW', 'CW'], rotation=30, ha="right", rotation_mode="anchor")

    ax2.set_yticks(np.arange(states_number))
    ax2.set_title(f'Action matrix\n{n_simulations} simulations', fontsize=7)
    ax2.set_xlabel('Action')
    ax2.set_ylabel('State')

    ax3.imshow(np.reshape([initial_state_list_distri],(-1,1)), vmin=0, vmax=1)
    ax3.set_xticks([])

    ax3.set_yticks(np.arange(states_number))
    ax3.set_title(f'Initial distribution', fontsize=7)
