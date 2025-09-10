#######################
### Import packages ###
#######################

from hmm_functions import *
import matplotlib.pyplot as plt
from hmmlearn import hmm, vhmm
import sys
import time
from IPython.core import ultratb
import dill
import numpy as np
sys.excepthook = ultratb.FormattedTB(call_pdb=False)

plt.style.use('paper.mplstyle')

start_time = time.time()

with open(f'DDM/synthetic_data.pkl', 'rb') as file:
    synthetic_data = dill.load(file)

training_data = [synthetic_data[i]['choices'] for i in np.arange(0,10)]
validation_data = [synthetic_data[i]['choices'] for i in np.arange(10,20)]


#training_mice_ordered_actions_types_number = [extract_actions_sequence(path_to_data_folder, mouse, session_index)[0] for mouse in training_mice]
#validation_mice_ordered_actions_types_number = [extract_actions_sequence(path_to_data_folder, mouse, session_index)[0] for mouse in validation_mice]

###################
### Infer model ###
###################

## Reformating data

training_emissions = np.array([]).astype(int)
validation_emissions = np.array([]).astype(int)

for x,y in zip(training_data,validation_data):

    training_emissions = np.concatenate((training_emissions, x))
    validation_emissions = np.concatenate((validation_emissions, y))

training_emissions = training_emissions.reshape(-1,1)
training_emissions_lengths = [len(x) for x in training_data]

validation_emissions = validation_emissions.reshape(-1,1)
validation_emissions_lengths = [len(y) for y in validation_data]

## Infer best model

# best_model, best_score = infer_best_model_score(training_emissions, validation_emissions, 
#                                           training_emissions_lengths, validation_emissions_lengths, 
#                                           [2,3,4,5,6,7,8,9], n_features=4, seed=13)

n_to_test = [2,3,4,5,6,7,8,9,10]

best_model, best_score = infer_best_model_aic(training_emissions, validation_emissions, 
                                          training_emissions_lengths, validation_emissions_lengths, 
                                          n_to_test, seed=13)

# best_model, best_score = infer_best_model_variational(training_emissions, validation_emissions, 
#                                           training_emissions_lengths, validation_emissions_lengths, 
#                                           n_to_test, seed=13)

###########################
### Save model and sets ###
###########################

with open(f'DDM/best_model_aic.pkl', 'wb') as file:
    dill.dump(best_model, file)

#with open(f'HMM/training_set_aic.pkl', 'wb') as file:
#    dill.dump([training_mice, training_mice_ordered_actions_types_number], file)

#with open(f'HMM/validation_set_aic.pkl', 'wb') as file:
#    dill.dump([validation_mice, validation_mice_ordered_actions_types_number], file)

# with open(f'HMM/best_model_aic_4s_{n_to_test[-1]}_type{type_number}_v1.pkl', 'wb') as file:
#     dill.dump(best_model, file)

# with open(f'HMM/training_set_aic_4s_{n_to_test[-1]}_type{type_number}_v1.pkl', 'wb') as file:
#     dill.dump([training_mice, training_mice_ordered_actions_types_number], file)

# with open(f'HMM/validation_set_aic_4s_{n_to_test[-1]}_type{type_number}_v1.pkl', 'wb') as file:
#     dill.dump([validation_mice, validation_mice_ordered_actions_types_number], file)


# with open(f'HMM/best_model_variational_{n_to_test[-1]}_type{type_number}.pkl', 'wb') as file:
#     dill.dump(best_model, file)

# with open(f'HMM/training_set_variational_{n_to_test[-1]}_type{type_number}.pkl', 'wb') as file:
#     dill.dump([training_mice, training_mice_ordered_actions_types_number], file)

# with open(f'HMM/validation_set_variational_{n_to_test[-1]}_type{type_number}.pkl', 'wb') as file:
#     dill.dump([validation_mice, validation_mice_ordered_actions_types_number], file)

print(f'Best score:      {best_score}')

print(f'Transmission Matrix Recovered:\n{best_model.transmat_.round(3)}\n\n')

print(f'Emission Matrix Recovered:\n{best_model.emissionprob_.round(3)}\n\n')

end_time = time.time()

print(f"Ca a pris {(end_time-start_time)//60} min {(end_time-start_time)%60} s")


