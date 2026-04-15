###############
### Imports ###
###############

import matplotlib.pyplot as plt
import pickle
import numpy as np
import joblib as jl
from tqdm import tqdm
import os

from functions_synthetic_data_generation import *
from functions_synthetic_data_analysis import *
from functions_hmm import *

plt.style.use('paper.mplstyle')

##################
### Parameters ### 
##################

# Meta-parameters
MAIN_FOLDER_PATH = '/home/david/Documents/code/loop_noiseless_drift_estimation_output'

# Simulation generation parameters

P_CW_REWARD = 0.8
P_CCW_REWARD = 0
P_CW_INIT_RANGE = np.linspace(0.01,0.99,100)
STEPS_NUMBER = 40
NOISE_AMPLITUDE = 0
DRIFT_INIT = 0
DRIFT_VALUES_ARR = np.linspace(0.005,0.1,5)

DEFAULT_ARGS_DICT = {'p_cw_reward': P_CW_REWARD, 
             'p_ccw_reward': P_CCW_REWARD, 
             'p_cw_init': 0.5, 
             'steps_number': STEPS_NUMBER, 
             'noise_amplitude': NOISE_AMPLITUDE, 
             'drift_matrix': np.array([[0.1  ,  -0.1 ],
                                       [-0.05, 0.05]]), 
             'drift_init': DRIFT_INIT}


SIMULATIONS_SET_SIZE = 50
N_SETS = 100
SIMULATIONS_SET_SIZE_LIST = [SIMULATIONS_SET_SIZE]*N_SETS

# Fit parameters
STATES_NUMBER_RANGE = np.arange(2,12)
N_FITS = 50
N_JOBS = 5

#################################
### Synthetic data generation ###
#################################

# User decides whether or not to generate new simulations
answer = input('Do you want to generate new simulations sets ? [y,n]: ')

if answer=='y':
    generate_new_synthetic_data = True
elif answer=='n':
    generate_new_synthetic_data = False
else:
    print('ERROR: invalid input for whether to generate new simulations of not.\nShould either y or n.')
    quit()

if generate_new_synthetic_data:

    for n, drift in enumerate(DRIFT_VALUES_ARR):

        if n!=2:
            continue

        temp_args_dict = copy.deepcopy(DEFAULT_ARGS_DICT)
        # temp_args_dict['drift_matrix'] = np.array([[drift, -drift],
        #                                            [0    , 0     ]])

        for i, n_simulations in enumerate(SIMULATIONS_SET_SIZE_LIST):
                            
            simulations_set = run_simulations_batch_random_init(temp_args_dict, 
                                                                n_simulations, 
                                                                P_CW_INIT_RANGE)

            with open(f'{MAIN_FOLDER_PATH}/drift_{n}/simulation_set_{i}.pkl', 'wb') as file:
                pickle.dump(simulations_set, file)

hipitihopitiyourcodewillnowstopiti

# Fit HMMs to data

for n, drift in tqdm(enumerate(DRIFT_VALUES_ARR),desc='Drift processed',leave=False):

    for i in tqdm(range(N_SETS),desc='Sets processed', leave=False):

        n_simulations = SIMULATIONS_SET_SIZE_LIST[i]

        with open(f'{MAIN_FOLDER_PATH}/drift_{n}/simulation_set_{i}.pkl', 'rb') as file:
            synthetic_data = pickle.load(file)

        reformated_training_sequences, reformated_validation_sequences, reformated_training_sequences_lengths, reformated_validation_sequences_lengths = reformat_synthetic_data(synthetic_data)

        fit_output = infer_best_model_score_parallel(reformated_training_sequences, 
                                                    reformated_validation_sequences, 
                                                    reformated_training_sequences_lengths, 
                                                    reformated_validation_sequences_lengths, 
                                                    STATES_NUMBER_RANGE, 
                                                    fix_probability=True, 
                                                    n_features = None, 
                                                    n_fits=50, 
                                                    n_jobs=4)

        save_path = f'{MAIN_FOLDER_PATH}/drift_{n}/hmm_fit_output_{i}.pkl'

        with open(save_path, 'wb') as file:
            pickle.dump(fit_output, file)

        