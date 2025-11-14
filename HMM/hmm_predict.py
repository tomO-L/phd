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
path_to_data_folder='/LocalData/ForagingMice/4TowersTaskMethodPaper_Data/AurelienData'
path_to_data_folder='/LocalData/ForagingMice/4TowersTaskMethodPaper_Data/MaudData'

# Analysing the entire group of mice
mice_sample_group_1 = [
    "MOUEml1_5", "MOUEml1_8", "MOUEml1_11", "MOUEml1_12", "MOUEml1_13", "MOUEml1_15", "MOUEml1_18", "MOUEml1_20",
    "MOURhoA_2", "MOURhoA_5", "MOURhoA_6", "MOURhoA_8", "MOURhoA_9", "MOURhoA_12", "MOURhoA_14",
    "MOUB6NN_4", "MOUB6NN_6", "MOUB6NN_13", "MOUB6NN_15"
]

mice_sample_group_2 = ['MOU3974','MOU3975', 'MOU3987', 'MOU3988', 'MOU3991', 'MOU3992', 'MOU4551', 'MOU4552', 'MOU4560', 'MOU4561', 'MOU4562',
               'MOU4563', 'MOU4623', 'MOU4964', 'MOU4965', 'MOU4986', 'MOU4987', 'MOU4988', 'MOU4993', 'MOU5007', 'MOU5008']

test_mice_group_1 = mice_sample_group_1

test_mice_group_2 = ['MOU4993',
             'MOU3974',
             'MOU3992',
             'MOU4987',
             'MOU4988'
            ]

test_mice = test_mice_group_2

### Test mice ###

## CW
# MOU4993

## CCW
# MOU3974
# MOU3992
# MOU4987 
# MOU4988

mice_to_analyse = test_mice

sessions_index = np.arange(0,20)
nb_of_sessions = len(sessions_index)

mouse_index = 2
mouse_name = mice_to_analyse[mouse_index]

folder_path_mouse_to_analyse = os.path.join(path_to_data_folder,mouse_name)

#####################
### Loading Model ###
#####################

type_number = 2

# best_model = f'best_model_aic_10_type{type_number}'
# training_set = f'training_set_aic_10_type{type_number}'
# validation_set = f'validation_set_aic_10_type{type_number}'

# model = f'best_model_aic_10_type{type_number}'
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

mice_rewarded_tats_persession, mice_unrewarded_tats_persession = compute_rewards_persession(path_to_data_folder,[mouse_name])

states_sequences = []

for session_index in sessions_index:
    
    mouse_actions_sequence = extract_actions_sequence(path_to_data_folder, mouse_name, session_index)[0]
    mouse_num_runs = len(mouse_actions_sequence)
    
    states_sequence = model.predict(np.int16(mouse_actions_sequence.reshape(-1,1)))
    states_sequences.append(states_sequence)

states_distributions = compute_states_distribution_persession(path_to_data_folder,mouse_name,sessions_index,model)
states_time_distributions, states_time_ratio_distributions = compute_time_in_states_persession(path_to_data_folder,mouse_name,sessions_index,model)

action_types = extract_actions_sequence(path_to_data_folder, mouse_name, 0)[2]

#############
### Plots ###
#############

output_path = 'HMM/figures/'

colors= ['#d62728','#7f7f7f','#bcbd22','#2ca02c','#1f77b4','#ff7f0e']
# colors = ['purple','green','yellow','red','maroon','blue','grey','orange']

### States Sequences ###

max_length = 0

for states_sequence in states_sequences:

    max_length = max(max_length, len(states_sequence))

padded_states_sequences = []

for states_sequence in states_sequences:

    padded_states_sequence = np.pad(np.array(states_sequence, dtype=float), (0,max_length-len(states_sequence)+1), mode='constant', constant_values=(np.nan,np.nan))
    padded_states_sequences.append(padded_states_sequence)

fig=plt.figure(figsize=(7, 4), dpi=300, constrained_layout=False, facecolor='w')
gs = fig.add_gridspec(1, 1)
row = gs[0,0].subgridspec(1, 1)
ax = plt.subplot(row[0,0])
    
plot_states_sequence(ax, padded_states_sequences, colors=colors)

ax.set_yticks(np.arange(nb_of_sessions),np.arange(nb_of_sessions)+1)
ax.set_xticks(np.arange(0,max_length,50))

ax.set_title(mouse_name, y=1.3)

fig.tight_layout()
fig.savefig(f"{output_path}{mouse_name}_states_sequences.png", facecolor='w', edgecolor='none', bbox_inches='tight', format="png", dpi=300)

### State distribution ###


## One session

# fig=plt.figure(figsize=(4, 7), dpi=300, constrained_layout=False, facecolor='w')
# gs = fig.add_gridspec(1, 1)
# row = gs[0,0].subgridspec(1, 1)
# ax = plt.subplot(row[0,0])

# plot_states_distribution(states_sequence,ax)


## Across sessions
fig=plt.figure(figsize=(7, 4), dpi=300, constrained_layout=False, facecolor='w')
gs = fig.add_gridspec(1, 1)
row = gs[0,0].subgridspec(1, 1)
ax = plt.subplot(row[0,0])
ax_bis = ax.twinx()

# plot_states_distri_across_sessions(states_distributions, ax)

# plot_states_distri_across_sessions(states_time_distributions/25, ax)


plot_states_distri_across_sessions(states_time_ratio_distributions, ax, colors=colors)

plot_individual_learning_curve(mice_rewarded_tats_persession, ax_bis, sessions_range=[0,20])
plot_individual_learning_curve(mice_unrewarded_tats_persession, ax_bis, sessions_range=[0,20], linestyle='--')

ax.set_title(mouse_name)

fig.savefig(f"{output_path}{mouse_name}_states_time_ratio.png", facecolor='w', edgecolor='none', bbox_inches='tight', format="png", dpi=300)

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
# ax6.set_xticks(range(len(action_types)), labels=action_types, rotation=30, ha="right", rotation_mode="anchor")
ax6.set_xticks(range(len(action_types)), labels=['CW Quarter Turn', 'CCW Quarter Turn', 'Between Towers', 'Toward Tower', 'Exploratory'], rotation=30, ha="right", rotation_mode="anchor")
ax6.set_yticks(np.arange(nb_of_states))
ax6.set_title('Emission matrix')
ax6.set_xlabel('Action')
ax6.set_ylabel('State')

### Test Plots ###
example_session_index = 16

all_epochs = load_pickle_data(folder_path_mouse_to_analyse, example_session_index)["all_epochs"]
ordered_runs = order_runs(all_epochs)[0]

example_states_sequence = states_sequences[example_session_index]

window_size = 50

## 
fig=plt.figure(figsize=(9, 3), dpi=300, constrained_layout=False, facecolor='w')
gs = fig.add_gridspec(1, 1)
row = gs[0,0].subgridspec(1, 1)
ax = plt.subplot(row[0,0])

plot_cumulated_turns_profile(ordered_runs, ax, states_sequence = example_states_sequence, colors = colors)

ax.set_xlabel('Run rank')
ax.set_yticks([])
ax.set_yticklabels([])

## 
fig=plt.figure(figsize=(7, 4), dpi=300, constrained_layout=False, facecolor='w')
gs = fig.add_gridspec(1, 1)
row = gs[0,0].subgridspec(1, 1)
ax = plt.subplot(row[0,0])

plot_cumulated_states_profile(example_states_sequence, np.arange(nb_of_states), ax, colors = colors)

## 
fig=plt.figure(figsize=(7, 4), dpi=300, constrained_layout=False, facecolor='w')
gs = fig.add_gridspec(1, 1)
row = gs[0,0].subgridspec(1, 1)
ax = plt.subplot(row[0,0])

plot_reward_rate(ordered_runs, ax, window_size=window_size)
plot_states_occurence_frequency(example_states_sequence, np.arange(nb_of_states), ax, colors = colors, window_size=window_size)

## 

############################################################################
def plot_actions_sequence(ax, ordered_actions_types_number, ordered_actions_frames=[], 
                       action_types = ['run_around_tower', 'run_between_towers', 'run_toward_tower', 'exploratory_run'], show_yticks=True):

    num_actions = len(ordered_actions_types_number)

    for i in range(len(action_types)):


        if len(ordered_actions_frames)==0:

            ordered_actions_type_number = np.where(np.array(ordered_actions_types_number)==i, np.arange(num_actions)-0.25, np.nan)
            x_barh = np.transpose([ordered_actions_type_number,0.5*np.ones(num_actions)])
        
        else:

            ordered_actions_frames = np.array(ordered_actions_frames)
            ordered_actions_frame = np.where(np.array(ordered_actions_types_number)==i, ordered_actions_frames[:,0], np.nan)
            ordered_actions_width = np.where(np.array(ordered_actions_types_number)==i, ordered_actions_frames[:,1] - ordered_actions_frames[:,0], np.nan)

            x_barh = np.transpose([ordered_actions_frame,ordered_actions_width])

        y_barh = [i-0.25,0.5]

        ax.broken_barh(x_barh, y_barh, facecolor='black')

    if show_yticks:
        ax.set_yticks(np.arange(len(action_types)), ['CW Quarter Turn', 'CCW Quarter Turn', 'Between Towers', 'Toward Tower', 'Exploratory'])

def plot_states_sequence_v2(ax, states_sequence, colors = ['#d62728','#7f7f7f','#bcbd22','#2ca02c','#1f77b4','#ff7f0e']):

    for i in range(len(ordered_runs)):

        ax.axvspan(i-0.5,i+0.5, color=colors[states_sequence[i]], ymin=0., ymax=1, alpha=0.75, linewidth=0, zorder=0)

    cmap = ListedColormap(colors) #plt.cm.Set1
    norm = Normalize(vmin=np.nanmin(states_sequence), vmax=np.nanmax(states_sequence)+1)

    ticks = np.arange(np.nanmax(states_sequence)+1)

    fig = ax.get_figure()

    cbar = fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap), ax=ax, drawedges=False, orientation='horizontal', label='States', location='top', alpha=0.75)
    cbar.set_ticks(ticks=ticks+0.5, labels=np.int8(ticks))

    ax.set_xlabel('Run rank')

############################################################################

fig=plt.figure(figsize=(7, 4), dpi=300, constrained_layout=False, facecolor='w')
gs = fig.add_gridspec(1, 1)
row = gs[0,0].subgridspec(1, 1)
ax = plt.subplot(row[0,0])

actions_sequence = extract_actions_sequence(path_to_data_folder, mouse_name, example_session_index)[0]

plot_states_sequence_v2(ax, example_states_sequence)

plot_actions_sequence(ax, actions_sequence, action_types=action_types)

ax.set_xlim([-0.5,300.5])
ax.set_ylim([None,5])

plt.show()

