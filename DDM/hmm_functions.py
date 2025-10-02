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
            
            model.fit(x_train, training_lengths)
            score = model.score(x_validate, validation_lengths)
            # print(f'Model {n} components #{idx}\tScore: {score}')
            if best_score is None or score > best_score:
                best_model = model
                best_score = score

    return best_model, best_score

# def infer_best_model(x_train, x_validate, lengths, n_to_test, n_features = 4, seed=13):
#     # check optimal score

#     best_score = best_model = None
#     n_fits = 200
#     np.random.seed(seed)

#     for n in n_to_test:
#         print(f"Building {n} components model")
#         for idx in tqdm(range(n_fits)):
#             # model = hmm.CategoricalHMM(
#             model = vhmm.VariationalCategoricalHMM(
#                 n_components=n, random_state=idx,
#                 init_params='ste', algorithm='viterbi', n_features=n_features)  # don't init transition, set it below
#             # we need to initialize with random transition matrix probabilities
#             # because the default is an even likelihood transition
#             # we know transitions are rare (otherwise the casino would get caught!)
#             # so let's have an Dirichlet random prior with an alpha value of
#             # (0.1, 0.9) to enforce our assumption transitions happen roughly 10%
#             # of the time

#             # transmat = []
#             # for _ in range(n):

#             #     row = np.random.uniform(size=n)
#             #     row = row/np.sum(row)
                
#             #     transmat.append(row)

#             # transmat = np.array(transmat)
#             # model.transmat_ = transmat
            
#             model.fit(x_train, lengths)
#             score = model.score(x_validate)
#             # print(f'Model {n} components #{idx}\tScore: {score}')
#             if best_score is None or score > best_score:
#                 best_model = model
#                 best_score = score

#     return best_model, best_score
