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
sys.excepthook = ultratb.FormattedTB(call_pdb=False)

plt.style.use('paper.mplstyle')

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

####################
### Loading data ###
####################

type_number = 3

best_model = f'best_model_aic_10_type{type_number}'
training_set = f'training_set_aic_10_type{type_number}'
validation_set = f'validation_set_aic_10_type{type_number}'

with open(f'HMM/{best_model}.pkl', 'rb') as file:
    best_model = dill.load(file)

with open(f'HMM/{training_set}.pkl', 'rb') as file:
    training_mice, training_mice_ordered_epochs_types_number = dill.load(file)

with open(f'HMM/{validation_set}.pkl', 'rb') as file:
    validation_mice, validation_mice_ordered_epochs_types_number = dill.load(file)

print(f'Transmission Matrix Recovered:\n{best_model.transmat_.round(3)}\n\n')

print(f'Emission Matrix Recovered:\n{best_model.emissionprob_.round(3)}\n\n')

#############
### Plots ###
#############



fig=plt.figure(figsize=(4, 7), dpi=300, constrained_layout=False, facecolor='w')
gs = fig.add_gridspec(2, 1, hspace=0.5, height_ratios=[0.3,1])
row = gs[0,0].subgridspec(1, 1)

ax1 = plt.subplot(row[0,0])

example_mouse_index = 0
example_mouse_name = validation_mice[example_mouse_index]
example_mouse_num_runs = len(validation_mice_ordered_epochs_types_number[example_mouse_index])
action_types = extract_actions_sequence(path_to_data_folder, example_mouse_name, example_mouse_index)[2]
plot_actions_distribution(ax1, validation_mice_ordered_epochs_types_number[example_mouse_index], action_types=action_types)

row_bis = gs[1,0].subgridspec(2, 1, height_ratios=[0.25,1], hspace=0.1)

ax2 = plt.subplot(row_bis[0,0])
ax3 = plt.subplot(row_bis[1,0])

states = best_model.predict(np.int16(validation_mice_ordered_epochs_types_number[0].reshape(-1,1)))
plot_states_sequence(ax2, states)

plot_actions_sequence(ax3, validation_mice_ordered_epochs_types_number[example_mouse_index], action_types=action_types)


# plot generated states

# fig, ax = plt.subplots()
# ax.plot(abs(gen_states), label='generated')
# ax.set_yticks(list(set(states)))
# ax.set_title('Generated states')
# ax.set_xlabel('Run rank')
# ax.set_ylabel('State')
# ax.legend()

# plot matrices

fig=plt.figure(figsize=(3.5, 3), dpi=300, constrained_layout=False, facecolor='w')
gs = fig.add_gridspec(1, 1)
row = gs[0].subgridspec(1,2)
ax5 = plt.subplot(row[0,0])
ax6 = plt.subplot(row[0,1])

nb_of_states = len(best_model.transmat_)

ax5.imshow(best_model.transmat_)
ax5.set_xticks(np.arange(nb_of_states))
ax5.set_yticks(np.arange(nb_of_states))

ax5.set_title('Transition matrix')
ax5.set_xlabel('To state')
ax5.set_ylabel('From state')


ax6.imshow(best_model.emissionprob_)
ax6.set_xticks(range(len(action_types)), labels=action_types, rotation=30, ha="right", rotation_mode="anchor")
ax6.set_yticks(np.arange(nb_of_states))
ax6.set_title('Emission matrix')
ax6.set_xlabel('Action')
ax6.set_ylabel('State')

plt.show()


