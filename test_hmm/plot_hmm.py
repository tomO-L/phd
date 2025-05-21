#######################
### Import packages ###
#######################

from functions import *
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
#path_to_data_folder='/LocalData/ForagingMice/4TowersTaskMethodPaper_Data/AurelienData/'
path_to_data_folder='/LocalData/ForagingMice/4TowersTaskMethodPaper_Data/MaudData/'

# Analysing the entire group of mice
# mice_to_analyse = [
#     "MOUEml1_5", "MOUEml1_8", "MOUEml1_11", "MOUEml1_12", "MOUEml1_13", "MOUEml1_15", "MOUEml1_18", "MOUEml1_20",
#     "MOURhoA_2", "MOURhoA_5", "MOURhoA_6", "MOURhoA_8", "MOURhoA_9", "MOURhoA_12", "MOURhoA_14",
#     "MOUB6NN_4", "MOUB6NN_6", "MOUB6NN_13", "MOUB6NN_15"
# ]

mice_to_analyse = ['MOU3974','MOU3975', 'MOU3987', 'MOU3988', 'MOU3991', 'MOU3992', 'MOU4551', 'MOU4552', 'MOU4560', 'MOU4561', 'MOU4562',
                   'MOU4563', 'MOU4623', 'MOU4964', 'MOU4965', 'MOU4986', 'MOU4987', 'MOU4988', 'MOU4993', 'MOU5007', 'MOU5008']

####################
### Loading data ###
####################

best_model = 'best_model'
training_set = 'training_set'
validation_set = 'validation_set'

with open(f'test_hmm/{best_model}.pkl', 'rb') as file:
    best_model = dill.load(file)

with open(f'test_hmm/{training_set}.pkl', 'rb') as file:
    training_mice, training_mice_ordered_epochs_types_number = dill.load(file)

with open(f'test_hmm/{validation_set}.pkl', 'rb') as file:
    validation_mice, validation_mice_ordered_epochs_types_number = dill.load(file)

print(f'Transmission Matrix Recovered:\n{best_model.transmat_.round(3)}\n\n')

print(f'Emission Matrix Recovered:\n{best_model.emissionprob_.round(3)}\n\n')

#############
### Plots ###
#############

# plot recovered and generated sequences

fig=plt.figure(figsize=(4, 7), dpi=300, constrained_layout=False, facecolor='w')
gs = fig.add_gridspec(1, 1)
row1 = gs[0].subgridspec(2, 2)
ax1 = plt.subplot(row1[0,0])
ax2 = plt.subplot(row1[1,0])
ax3 = plt.subplot(row1[0,1])
ax4 = plt.subplot(row1[1,1])

example_mouse_index = 0
example_mouse_num_runs = len(validation_mice_ordered_epochs_types_number[example_mouse_index])

plot_runs_distribution(ax1, validation_mice_ordered_epochs_types_number[example_mouse_index])
plot_runs_sequence(ax2, validation_mice_ordered_epochs_types_number[example_mouse_index])

gen_ordered_epochs_types_number, gen_states = best_model.sample(example_mouse_num_runs)

plot_runs_distribution(ax3, gen_ordered_epochs_types_number.reshape(1,-1)[example_mouse_index])
plot_runs_sequence(ax4, gen_ordered_epochs_types_number.reshape(1,-1)[example_mouse_index])

ax1.set_title('Validation')
ax3.set_title('Recovered')

states = best_model.predict(np.int8(validation_mice_ordered_epochs_types_number[0].reshape(-1,1)))

# plot recovered states

fig, ax = plt.subplots()
ax.plot(abs(states), label='recovered')
ax.set_yticks(range(len(set(states))))
ax.set_title('Recovered states')
ax.set_xlabel('Run rank')
ax.set_ylabel('State')
ax.legend()

# plot generated states

fig, ax = plt.subplots()
ax.plot(abs(gen_states), label='generated')
ax.set_yticks(range(len(set(states))))
ax.set_title('Generated states')
ax.set_xlabel('Run rank')
ax.set_ylabel('State')
ax.legend()


# plot matrices

epoch_types = ['run_around_tower', 'run_between_towers', 'run_toward_tower', 'exploratory_run']

fig=plt.figure(figsize=(3.5, 3), dpi=300, constrained_layout=False, facecolor='w')
gs = fig.add_gridspec(1, 1)
row = gs[0].subgridspec(1,2)
ax5 = plt.subplot(row1[0,0])
ax6 = plt.subplot(row1[0,1])


ax5.imshow(best_model.transmat_)
ax5.set_title('Transition matrix')
ax5.set_xlabel('To state')
ax5.set_ylabel('From state')


ax6.imshow(best_model.emissionprob_)
ax6.set_xticks(range(len(epoch_types)), labels=epoch_types, rotation=30, ha="right", rotation_mode="anchor")
ax6.set_title('Emission matrix')
ax6.set_xlabel('Action')
ax6.set_ylabel('State')

plt.show()



