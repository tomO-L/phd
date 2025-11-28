import matplotlib.pyplot as plt
import dill
import numpy as np
# from tqdm import tqdm
from hmmlearn import hmm, vhmm

from synthetic_data_generation_functions import *
# from synthetic_data_analysis_functions import *
from hmm_functions import *


plt.style.use('/home/david/Documents/code/phd/paper.mplstyle')

steps_number = 100
noise_amplitude = 0.1
delta = 0.05
drift = 0.0
p_a = 0.5
p_a_reward = 0.8
p_b_reward = 0

number_of_simulations = 20

n_simulations_list = [number_of_simulations]*10


simulations_folder_path = '/home/david/Documents/code/DDM_v2_synthetic_data'


generate_simulations = False

for i, n_simulations in enumerate(n_simulations_list):
    
    if generate_simulations:

        break

    simulations_batch = run_simulations_batch(p_a, p_a_reward, p_b_reward, steps_number, noise_amplitude, delta, drift, n_simulations)

    with open(f'{simulations_folder_path}/n_{n_simulations}/simulations_batch_{n_simulations}_test_{i+1}.pkl', 'wb') as file:
        dill.dump(simulations_batch, file)

n_to_test = np.arange(2,16)

# for index, n_simulations in enumerate(n_simulations_list):
for index, n_simulations in enumerate(tqdm(n_simulations_list)):

    ####################
    ### Loading Data ###
    ####################

    with open(f'{simulations_folder_path}/n_{n_simulations}/simulations_batch_{n_simulations}_test_{index+1}.pkl', 'rb') as file:
        synthetic_data = dill.load(file)

    ########################
    ### Reformating Data ###
    ########################

    slice_size = int(n_simulations/2)

    training_data = [synthetic_data[i]['choices'] for i in np.arange(0,slice_size)]
    validation_data = [synthetic_data[i]['choices'] for i in np.arange(slice_size,2*slice_size)]


    training_emissions = np.array([]).astype(int)
    validation_emissions = np.array([]).astype(int)

    for x,y in zip(training_data,validation_data):

        training_emissions = np.concatenate((training_emissions, x))
        validation_emissions = np.concatenate((validation_emissions, y))

    training_emissions = training_emissions.reshape(-1,1)
    training_emissions_lengths = [len(x) for x in training_data]

    validation_emissions = validation_emissions.reshape(-1,1)
    validation_emissions_lengths = [len(y) for y in validation_data]

    ###################
    ### Infer model ###
    ###################

    best_model, best_score = infer_best_model_score(training_emissions, validation_emissions, 
                                            training_emissions_lengths, validation_emissions_lengths, 
                                            n_to_test, leave_loading_bar=False, verbose=False)
    
    ##################
    ### Save model ###
    ##################
    
    with open(f'{simulations_folder_path}/n_{n_simulations}/best_model_score_{n_simulations}_test_{index+1}.pkl', 'wb') as file:
        dill.dump(best_model, file)
