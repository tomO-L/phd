#######################
### Import packages ###
#######################

import numpy as np
import matplotlib.pyplot as plt
import joblib as jl
from hmmlearn import hmm, vhmm
import time
# from tqdm import tqdm
from tqdm.notebook import tqdm

def fit_hmm_fixed_states_number(x_train, x_validate, training_lengths, validation_lengths, n_states, n_fits=200, n_features=None, fix_probability=False):
                
    # for idx in tqdm(range(n_fits), leave=leave_loading_bar, desc=f'Building {n} components model'):
    
    for idx in range(n_fits):
        
        if not(fix_probability):
            
            model = hmm.CategoricalHMM(
                n_components=n_states, random_state=int(time.time()),
                init_params='ste', algorithm='viterbi', n_features=n_features)  # don't init transition, set it below

        else:

            model = hmm.CategoricalHMM(
                n_components=n_states, random_state=int(time.time()),
                init_params='st', params='st', algorithm='viterbi', n_features=n_features)  # don't init transition, set it below

            model.emissionprob_ = np.transpose(np.array([1-np.linspace(0,1,n_states),np.linspace(0,1,n_states)]))

        model.fit(x_train, training_lengths)
        score = model.score(x_validate, validation_lengths)

    return (model,score)

def infer_best_model_score_parallel(x_train, x_validate, training_lengths, validation_lengths, n_to_test, fix_probability=False, n_features = None, n_fits=200, n_jobs=-2):
    # check optimal score

    models_and_scores_list = jl.Parallel(n_jobs=n_jobs)(jl.delayed(fit_hmm_fixed_states_number)(x_train, x_validate, training_lengths, validation_lengths, n_states, 
                                                                                                fix_probability=fix_probability, n_fits=n_fits, n_features=n_features)
                                                                                                for n_states in tqdm(n_to_test))

    fit_output = {'models': [variable[0] for variable in models_and_scores_list], 'scores': [variable[1] for variable in models_and_scores_list]}

    return fit_output

def infer_best_model_score(x_train, x_validate, training_lengths, validation_lengths, n_to_test, fix_probability=False, n_features = None, n_fits=200, n_jobs=-2):
    # check optimal score
    models_and_scores_list = []

    for n_states in tqdm(n_to_test):

        res = fit_hmm_fixed_states_number(x_train, x_validate, training_lengths, validation_lengths, n_states, 
                                    fix_probability=fix_probability, n_fits=n_fits, n_features=n_features)
                                                        
        models_and_scores_list.append(res)

    fit_output = {'models': [variable[0] for variable in models_and_scores_list], 'scores': [variable[1] for variable in models_and_scores_list]}

    return fit_output

def reformat_synthetic_data(synthetic_data):

    slice_size = int(len(synthetic_data)/2)

    # synthetic_data = synthetic_data[:int(slice_size/2)] + synthetic_data[int(slice_size):int(3*slice_size/2)] + synthetic_data[int(slice_size/2):int(slice_size)] + synthetic_data[int(3*slice_size/2):]

    training_sequences = [synthetic_data[i]['choices'] for i in np.arange(0,slice_size)]
    validation_sequences = [synthetic_data[i]['choices'] for i in np.arange(slice_size,2*slice_size)]

    ###################
    ### Infer model ###
    ###################

    ## Reformating data

    reformated_training_sequences = np.array([]).astype(int)
    reformated_validation_sequences = np.array([]).astype(int)

    for x,y in zip(training_sequences,validation_sequences):

        reformated_training_sequences = np.concatenate((reformated_training_sequences, x))
        reformated_validation_sequences = np.concatenate((reformated_validation_sequences, y))

    reformated_training_sequences = reformated_training_sequences.reshape(-1,1)
    reformated_training_sequences_lengths = [len(x) for x in training_sequences]

    reformated_validation_sequences = reformated_validation_sequences.reshape(-1,1)
    reformated_validation_sequences_lengths = [len(y) for y in validation_sequences]

    return reformated_training_sequences, reformated_validation_sequences, reformated_training_sequences_lengths, reformated_validation_sequences_lengths

