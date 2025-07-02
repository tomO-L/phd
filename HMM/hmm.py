#######################
### Import packages ###
#######################

from hmm_functions import *
from actions_functions import *
from plots_functions import *
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

######################
### Mice selection ###
######################

# defining data folder path and mice list
# path_to_data_folder is the path of the folder where you store the folders of your different mice.
path_to_data_folder='/LocalData/ForagingMice/4TowersTaskMethodPaper_Data/AurelienData/'
# path_to_data_folder='/LocalData/ForagingMice/4TowersTaskMethodPaper_Data/MaudData/'

# Analysing the entire group of mice
mice_to_analyse = [
    "MOUEml1_5", "MOUEml1_8", "MOUEml1_11", "MOUEml1_12", "MOUEml1_13", "MOUEml1_15", "MOUEml1_18", "MOUEml1_20",
    "MOURhoA_2", "MOURhoA_5", "MOURhoA_6", "MOURhoA_8", "MOURhoA_9", "MOURhoA_12", "MOURhoA_14",
    "MOUB6NN_4", "MOUB6NN_6", "MOUB6NN_13", "MOUB6NN_15"
]

# mice_to_analyse = ['MOU3974','MOU3975', 'MOU3987', 'MOU3988', 'MOU3991', 'MOU3992', 'MOU4551', 'MOU4552', 'MOU4560', 'MOU4561', 'MOU4562',
#                    'MOU4563', 'MOU4623', 'MOU4964', 'MOU4965', 'MOU4986', 'MOU4987', 'MOU4988', 'MOU4993', 'MOU5007', 'MOU5008']

##################
### Parameters ###
##################

training_mice = mice_to_analyse[0:9]
validation_mice = mice_to_analyse[9:18]

session_index = 13

######################
### Extract actions ###
######################

training_mice_ordered_actions_types_number = [extract_actions_sequence(path_to_data_folder, mouse, session_index)[0] for mouse in training_mice]
validation_mice_ordered_actions_types_number = [extract_actions_sequence(path_to_data_folder, mouse, session_index)[0] for mouse in validation_mice]

###################
### Infer model ###
###################

# Reformating data

training_emissions = np.array([]).astype(int)
validation_emissions = np.array([]).astype(int)

for x,y in zip(training_mice_ordered_actions_types_number,validation_mice_ordered_actions_types_number):

    training_emissions = np.concatenate((training_emissions, x))
    validation_emissions = np.concatenate((validation_emissions, y))

training_emissions = training_emissions.reshape(-1,1)
training_emissions_lengths = [len(x) for x in training_mice_ordered_actions_types_number]

validation_emissions = validation_emissions.reshape(-1,1)
validation_emissions_lengths = [len(y) for y in validation_mice_ordered_actions_types_number]

# Infer best model

# best_model, best_score = infer_best_model_score(training_emissions, validation_emissions, 
#                                           training_emissions_lengths, validation_emissions_lengths, 
#                                           [2,3,4,5,6,7,8,9], n_features=4, seed=13)

n_to_test = [2,3,4,5,6,7,8,9,10]

best_model, best_score = infer_best_model_aic(training_emissions, validation_emissions, 
                                          training_emissions_lengths, validation_emissions_lengths, 
                                          n_to_test, seed=13)

type_number = 3

###########################
### Save model and sets ###
###########################

with open(f'HMM/best_model_aic_{n_to_test[-1]}_type{type_number}.pkl', 'wb') as file:
    dill.dump(best_model, file)

with open(f'HMM/training_set_aic_{n_to_test[-1]}_type{type_number}.pkl', 'wb') as file:
    dill.dump([training_mice, training_mice_ordered_actions_types_number], file)

with open(f'HMM/validation_set_aic_{n_to_test[-1]}_type{type_number}.pkl', 'wb') as file:
    dill.dump([validation_mice, validation_mice_ordered_actions_types_number], file)

print(f'Best score:      {best_score}')

print(f'Transmission Matrix Recovered:\n{best_model.transmat_.round(3)}\n\n')

print(f'Emission Matrix Recovered:\n{best_model.emissionprob_.round(3)}\n\n')

end_time = time.time()

print(f"Ca a pris {(end_time-start_time)//60} min {(end_time-start_time)%60} s")


