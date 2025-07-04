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
from analysis_functions import *

sys.excepthook = ultratb.FormattedTB(call_pdb=False)

plt.style.use('paper.mplstyle')

######################
### Mice selection ###
######################

# defining data folder path and mice list
# path_to_data_folder is the path of the folder where you store the folders of your different mice.
# path_to_data_folder='/LocalData/ForagingMice/4TowersTaskMethodPaper_Data/AurelienData/'
path_to_data_folder='/LocalData/ForagingMice/4TowersTaskMethodPaper_Data/MaudData/'

# Analysing the entire group of mice
# mice_sample = [
#     "MOUEml1_5", "MOUEml1_8", "MOUEml1_11", "MOUEml1_12", "MOUEml1_13", "MOUEml1_15", "MOUEml1_18", "MOUEml1_20",
#     "MOURhoA_2", "MOURhoA_5", "MOURhoA_6", "MOURhoA_8", "MOURhoA_9", "MOURhoA_12", "MOURhoA_14",
#     "MOUB6NN_4", "MOUB6NN_6", "MOUB6NN_13", "MOUB6NN_15"
# ]

mice_sample = ['MOU3974','MOU3975', 'MOU3987', 'MOU3988', 'MOU3991', 'MOU3992', 'MOU4551', 'MOU4552', 'MOU4560', 'MOU4561', 'MOU4562',
               'MOU4563', 'MOU4623', 'MOU4964', 'MOU4965', 'MOU4986', 'MOU4987', 'MOU4988', 'MOU4993', 'MOU5007', 'MOU5008']

test_mice = ['MOU4993',
             'MOU3974',
             'MOU3992',
             'MOU4987',
             'MOU4988'
            ]

### Test mice ###

## CW
# MOU4993

## CCW
# MOU3974
# MOU3992
# MOU4987 
# MOU4988

mice_to_analyse = test_mice

#####################
### Loading Model ###
#####################

type_number = 2

# best_model = f'best_model_aic_10_type{type_number}'
# training_set = f'training_set_aic_10_type{type_number}'
# validation_set = f'validation_set_aic_10_type{type_number}'

model = f'best_model_aic_10_type{type_number}'
# training_set = f'training_set_variational_10_type{type_number}'
# validation_set = f'validation_set_variational_10_type{type_number}'

with open(f'HMM/{model}.pkl', 'rb') as file:
    model = dill.load(file)

print(f'Transmission Matrix Recovered:\n{model.transmat_.round(3)}\n\n')

print(f'Emission Matrix Recovered:\n{model.emissionprob_.round(3)}\n\n')

###################
### Using model ###
###################

# session_index = 19

# mouse_index = 2
# mouse_name = mice_to_analyse[mouse_index]
# mouse_actions_sequence = extract_actions_sequence(path_to_data_folder, mouse_name, session_index)[0]
# mouse_num_runs = len(mouse_actions_sequence)

# action_types = extract_actions_sequence(path_to_data_folder, mouse_name, session_index)[2]


sessions_index = np.arange(0,20)
nb_of_sessions = len(sessions_index)

mouse_index = 4
mouse_name = mice_to_analyse[mouse_index]

states_sequences = []

for session_index in sessions_index:
    
    mouse_actions_sequence = extract_actions_sequence(path_to_data_folder, mouse_name, session_index)[0]
    mouse_num_runs = len(mouse_actions_sequence)
    
    states_sequence = model.predict(np.int16(mouse_actions_sequence.reshape(-1,1)))
    states_sequences.append(states_sequence)

states_distributions = compute_states_distribution_persession(path_to_data_folder,mouse_name,sessions_index,model)

action_types = extract_actions_sequence(path_to_data_folder, mouse_name, 0)[2]

#############
### Plots ###
#############

### States Sequences ###

fig=plt.figure(figsize=(4, 7), dpi=300, constrained_layout=False, facecolor='w')
gs = fig.add_gridspec(1, 1)
row = gs[0,0].subgridspec(1, 1)
ax = plt.subplot(row[0,0])

max_length = 0

for states_sequence in states_sequences:

    max_length = max(max_length, len(states_sequence))

padded_states_sequences = []

for states_sequence in states_sequences:

    padded_states_sequence = np.pad(np.array(states_sequence, dtype=float), (0,max_length-len(states_sequence)+1), mode='constant', constant_values=(np.nan,np.nan))
    padded_states_sequences.append(padded_states_sequence)
    
plot_states_sequence(ax, padded_states_sequences)

ax.set_yticks(np.arange(nb_of_sessions),np.arange(nb_of_sessions)+1)
ax.set_xticks(np.arange(0,max_length,50))

### State distribution ###

## One session

# fig=plt.figure(figsize=(4, 7), dpi=300, constrained_layout=False, facecolor='w')
# gs = fig.add_gridspec(1, 1)
# row = gs[0,0].subgridspec(1, 1)
# ax = plt.subplot(row[0,0])

# plot_states_distribution(states_sequence,ax)

## Across sessions
fig=plt.figure(figsize=(4, 7), dpi=300, constrained_layout=False, facecolor='w')
gs = fig.add_gridspec(1, 1)
row = gs[0,0].subgridspec(1, 1)
ax = plt.subplot(row[0,0])

plot_states_distri_across_sessions(states_distributions, ax)

### Action and transition matrixes ###

fig=plt.figure(figsize=(3.5, 3), dpi=300, constrained_layout=False, facecolor='w')
gs = fig.add_gridspec(1, 1)
row = gs[0].subgridspec(1,2)
ax5 = plt.subplot(row[0,0])
ax6 = plt.subplot(row[0,1])

nb_of_states = len(model.transmat_)

ax5.imshow(model.transmat_)
ax5.set_xticks(np.arange(nb_of_states))
ax5.set_yticks(np.arange(nb_of_states))

ax5.set_title('Transition matrix')
ax5.set_xlabel('To state')
ax5.set_ylabel('From state')


ax6.imshow(model.emissionprob_)
ax6.set_xticks(range(len(action_types)), labels=action_types, rotation=30, ha="right", rotation_mode="anchor")
ax6.set_yticks(np.arange(nb_of_states))
ax6.set_title('Emission matrix')
ax6.set_xlabel('Action')
ax6.set_ylabel('State')

plt.show()


