#######################
### Import packages ###
#######################

import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from hmmlearn import hmm, vhmm
from tqdm import tqdm

########################
### Define functions ###
########################

def infer_best_model_score(x_train, x_validate, training_lengths, validation_lengths, n_to_test, n_features = None, seed=13):
    # check optimal score

    best_score = best_model = None
    n_fits = 200
    np.random.seed(seed)

    local_best_score_list = [] ### VERBOSE

    for n in n_to_test:
        print(f"Building {n} components model")
        
        local_best_score = None ### VERBOSE
        
        for idx in tqdm(range(n_fits)):
            # model = hmm.CategoricalHMM(
            model = hmm.CategoricalHMM(
                n_components=n, random_state=idx,
                init_params='ste', algorithm='viterbi', n_features=n_features)  # don't init transition, set it below
            
            model.fit(x_train, training_lengths)
            score = model.score(x_validate, validation_lengths)
            # print(f'Model {n} components #{idx}\tScore: {score}')
            if best_score is None or score > best_score:
                best_model = model
                best_score = score

            if local_best_score is None or score > local_best_score: ### VERBOSE

                local_best_score = score ### VERBOSE

        local_best_score_list.append(local_best_score) ### VERBOSE

    print("All Scores : ", local_best_score_list) ### VERBOSE

    return best_model, best_score


def infer_best_model_aic(x_train, x_validate, training_lengths, validation_lengths, n_to_test, n_features = None, seed=13):
    # check optimal score

    best_score = best_model = None
    n_fits = 200
    model_list = []
    aic_list = []

    np.random.seed(seed)

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