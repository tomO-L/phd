#######################
### Import packages ###
#######################

import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import time
import joblib as jl
from hmmlearn import hmm, vhmm
# from tqdm import tqdm
from tqdm.notebook import tqdm

########################
### Define functions ###
########################

def fit_hmm_fixed_state_number(x_train, x_validate, training_lengths, validation_lengths, n_states, n_fits=200, n_features=None):
                
    # for idx in tqdm(range(n_fits), leave=leave_loading_bar, desc=f'Building {n} components model'):
    for idx in range(n_fits):
        
        model = hmm.CategoricalHMM(
            n_components=n_states, random_state=idx,
            init_params='ste', algorithm='viterbi', n_features=n_features)  # don't init transition, set it below
            
        model.fit(x_train, training_lengths)
        score = model.score(x_validate, validation_lengths)

    return (model,score)

def infer_best_model_score(x_train, x_validate, training_lengths, validation_lengths, n_to_test, n_features = None, n_fits=200, save_path=None):
    # check optimal score

    best_score = best_model = None
    n_fits = n_fits

    models_list = []
    scores_list = []
    
    for n in n_to_test:    
        
        # # for idx in tqdm(range(n_fits), leave=leave_loading_bar, desc=f'Building {n} components model'):
        # for idx in range(n_fits):
        
        #     model = hmm.CategoricalHMM(
        #         n_components=n, random_state=idx,
        #         init_params='ste', algorithm='viterbi', n_features=n_features)  # don't init transition, set it below
            
        #     model.fit(x_train, training_lengths)
        #     score = model.score(x_validate, validation_lengths)
        #     # print(f'Model {n} components #{idx}\tScore: {score}')
        #     if best_score is None or score > best_score:
        #         best_model = model
        #         best_score = score

        model,score = fit_hmm_fixed_state_number(x_train, x_validate, training_lengths, validation_lengths, n, n_fits=n_fits, n_features=n_features)

        models_list.append(model)
        scores_list.append(score)

        if best_score is None or score > best_score:
            best_model = model
            best_score = score

    if save_path:

        with open(save_path, 'wb') as file:
            pickle.dump((best_model, best_score, models_list, scores_list), file)



    return best_model, best_score, models_list, scores_list


def infer_best_model_score_parallel(x_train, x_validate, training_lengths, validation_lengths, n_to_test, n_features = None, n_fits=200, n_jobs=-2):
    # check optimal score

    n_fits = n_fits

    models_and_scores_list = jl.Parallel(n_jobs=n_jobs)(jl.delayed(infer_best_model_score_parallel)(x_train, x_validate, training_lengths, validation_lengths, n_states, n_fits=200, n_features=n_features) for n_states in n_to_test)

    return models_and_scores_list
































def infer_best_model_aic(x_train, x_validate, training_lengths, validation_lengths, n_to_test, n_features = None, n_fits=200):
    # check optimal score

    best_score = best_model = None
    model_list = []
    aic_list = []

    for n in n_to_test:
        print(f"Building {n} components model")
        for idx in tqdm(range(n_fits)):
            # model = hmm.CategoricalHMM(
            model = hmm.CategoricalHMM(
                n_components=n, random_state=idx,
                init_params='ste', algorithm='viterbi', n_features=n_features)  # don't init transition, set it below
            
            model.fit(x_train, training_lengths)
            score = model.score(x_validate, validation_lengths)
            # print(f'Model {n} components #{idx}\tScore: {score}')
            if best_score is None or score > best_score:
                local_best_model = model
                best_score = score
                
        print(best_score)
        model_list.append(local_best_model)
        aic_list.append(local_best_model.aic(x_validate,lengths=validation_lengths))

    min_aic = min(aic_list)
    min_idx = aic_list.index(min_aic)

    best_model = model_list[min_idx]

    return best_model, min_aic


def infer_best_model_variational(x_train, x_validate, training_lengths, validation_lengths, n_to_test, n_features = None, seed=13):
    # check optimal score

    best_score = best_model = None
    n_fits = 200
    np.random.seed(seed)

    for n in n_to_test:
        print(f"Building {n} components model")
        for idx in tqdm(range(n_fits)):
            # model = hmm.CategoricalHMM(
            model = vhmm.VariationalCategoricalHMM(
                n_components=n, random_state=idx,
                init_params='ste', algorithm='viterbi', n_features=n_features)  # don't init transition, set it below
            
            model.fit(x_train, training_lengths)
            score = model.score(x_validate, validation_lengths)
            # print(f'Model {n} components #{idx}\tScore: {score}')
            if best_score is None or score > best_score:
                best_model = model
                best_score = score

    return best_model, best_score

# def order_matrix(matrix, indexes):

#     temp_matrix = []

#     for i in indexes:
#         temp_matrix.append(matrix[:,i])

#     temp_matrix = np.array(temp_matrix)

#     new_matrix = []

#     for i in indexes:
#         new_matrix.append(temp_matrix[i,:])

#     new_matrix = np.transpose(new_matrix)

#     return new_matrix

def order_matrix(matrix, indexes):

    new_matrix = np.copy(matrix)

    length = len(matrix)

    new_matrix[np.arange(length)] = new_matrix[indexes]

    new_matrix[:,np.arange(length)] = new_matrix[:,indexes]

    return new_matrix