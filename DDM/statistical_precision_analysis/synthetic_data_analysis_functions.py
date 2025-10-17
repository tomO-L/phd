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
from synthetic_data_generation_functions import *
sys.excepthook = ultratb.FormattedTB(call_pdb=False)

# plt.style.use('paper.mplstyle')

# Time counter
start_time = time.time()

##################
### Parameters ###
##################



# np.random.seed(58777) # initial seed
# np.random.seed(587) # test seed
# np.random.seed(50) # test seed

#################
### Functions ###
#################


def compute_simulations_average(p_a, p_a_reward, steps_number, noise_amplitude, delta, drift, n_simulations=300):

    synthetic_proba_list = []

    for _ in tqdm(range(n_simulations), leave=False):
    
        ddm_result = run_simulation(p_a, p_a_reward, steps_number, noise_amplitude, delta, drift)

        synthetic_proba_list.append(ddm_result['p_a'])

    average_trajectory = np.mean(synthetic_proba_list,axis=0)

    return average_trajectory

def compute_mean_square_error(delta, args):
    
    p_a = args[0]
    p_a_reward = args[1]
    steps_number = args[2]
    noise_amplitude = args[3]
    drift = args[4]
    n_simulations = args[5]
    reconstructed_average_trajectory = args[6]

    average_trajectory = compute_simulations_average(p_a, p_a_reward, steps_number, noise_amplitude, delta, drift, n_simulations=n_simulations)

    mse = (np.sum((average_trajectory - reconstructed_average_trajectory))**2)/steps_number

    return mse

def compute_mean_square_error_opt(delta, args):
    
    p_a = args[0]
    p_a_reward = args[1]
    steps_number = args[2]
    noise_amplitude = args[3]
    drift = args[4]
    n_simulations = args[5]
    reconstructed_average_trajectory = args[6]

    average_trajectory = compute_simulations_average(p_a, p_a_reward, steps_number, noise_amplitude, delta[0], drift, n_simulations=n_simulations)

    mse = (np.sum((average_trajectory - reconstructed_average_trajectory))**2)/steps_number

    return mse

def compute_reconstructed_proba_sequence(choices_sequence, model):

    states_sequence = model.predict(np.int16(choices_sequence.reshape(-1,1)))

    emissionprob = model.emissionprob_

    reconstructed_p_a_sequence = []

    for s in states_sequence:

        reconstructed_p_a_sequence.append(emissionprob[s][1])

    return reconstructed_p_a_sequence

def compute_reconstructed_average_proba_sequence(choices_sequences, model):

    sequences_number = len(choices_sequences)
    reconstructed_p_a_sequences = []

    for i in range(sequences_number):
    
        choices_sequence = choices_sequences[i]
    
        reconstructed_p_a_sequence = compute_reconstructed_proba_sequence(choices_sequence,model)

        reconstructed_p_a_sequences.append(reconstructed_p_a_sequence)

    reconstructed_average_p_a = np.mean(reconstructed_p_a_sequences,axis=0)

    return reconstructed_average_p_a


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

def compute_reconstructed_average_proba_sequences_fulltraining(n_simulations_list):

    reconstructed_average_proba_sequences = []

    for n_simulations in n_simulations_list:
        
        with open(f'DDM/statistical_precision_analysis/simulations_batches/simulations_batch_{n_simulations}.pkl', 'rb') as file:
            synthetic_data = dill.load(file)

        test_data = [synth_data['choices'] for synth_data in synthetic_data]

        with open(f'DDM/statistical_precision_analysis/simulations_batches/best_model_score_{n_simulations}_fulltraining.pkl', 'rb') as file:
            model = dill.load(file)

        reconstructed_average_proba_sequences.append(compute_reconstructed_average_proba_sequence(test_data, model))

    return reconstructed_average_proba_sequences
