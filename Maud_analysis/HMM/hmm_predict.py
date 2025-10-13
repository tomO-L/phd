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
path_to_data_folder='/LocalData/ForagingMice/4TowersTaskMethodPaper_Data/Group2Data'

# Analysing the entire group of mice

mice_sample_group_2 = ['MOU3974','MOU3975', 'MOU3987', 'MOU3988', 'MOU3991', 'MOU3992', 'MOU4551', 'MOU4552', 'MOU4560', 'MOU4561', 'MOU4562',
               'MOU4563', 'MOU4623', 'MOU4964', 'MOU4965', 'MOU4986', 'MOU4987', 'MOU4988', 'MOU4993', 'MOU5007', 'MOU5008']

test_mice_group_1 = mice_sample_group_2

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
model = f'best_model_score_19_type{type_number}_v1'

# training_set = f'training_set_variational_10_type{type_number}'
# validation_set = f'validation_set_variational_10_type{type_number}'

with open(f'Maud_analysis/HMM/{model}.pkl', 'rb') as file:
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


# plot_states_distri_across_sessions(states_time_ratio_distributions, ax, colors=colors)

# plot_individual_learning_curve(mice_rewarded_tats_persession, ax_bis, sessions_range=[0,20])
# plot_individual_learning_curve(mice_unrewarded_tats_persession, ax_bis, sessions_range=[0,20], linestyle='--')

ax.set_title(mouse_name)

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

### Test Plots ###
example_session_index = 16

all_epochs = load_pickle_data(folder_path_mouse_to_analyse, example_session_index)["all_epochs"]
ordered_runs = order_runs(all_epochs)[0]

example_states_sequence = states_sequences[example_session_index]

window_size = 50

## 
fig=plt.figure(figsize=(7, 4), dpi=300, constrained_layout=False, facecolor='w')
gs = fig.add_gridspec(1, 1)
row = gs[0,0].subgridspec(1, 1)
ax = plt.subplot(row[0,0])

# plot_cumulated_turns_profile(ordered_runs, ax, states_sequence = example_states_sequence, colors = colors)

## 
fig=plt.figure(figsize=(7, 4), dpi=300, constrained_layout=False, facecolor='w')
gs = fig.add_gridspec(1, 1)
row = gs[0,0].subgridspec(1, 1)
ax = plt.subplot(row[0,0])

# plot_cumulated_states_profile(example_states_sequence, np.arange(nb_of_states), ax, colors = colors)

## 
fig=plt.figure(figsize=(7, 4), dpi=300, constrained_layout=False, facecolor='w')
gs = fig.add_gridspec(1, 1)
row = gs[0,0].subgridspec(1, 1)
ax = plt.subplot(row[0,0])

plot_reward_rate(ordered_runs, ax, window_size=window_size)
# plot_states_occurence_frequency(example_states_sequence, np.arange(nb_of_states), ax, colors = colors, window_size=window_size)

### Reconstructed Probabilities ###

emissionprob = model.emissionprob_

# reconstructed_p_a_sequences = []

# for states_seq in states_sequences:

#     reconstructed_p_a_seq = []

#     for s in states_seq:

#         reconstructed_p_a_seq.append(emissionprob[s][1])

#     reconstructed_p_a_sequences.append(reconstructed_p_a_seq)

# p_a_sequences = [synth_data['p_a'] for synth_data in synthetic_data]

fig=plt.figure(figsize=(7, 4), dpi=300, constrained_layout=False, facecolor='w')
gs = fig.add_gridspec(1, 1)
row = gs[0,0].subgridspec(1, 1)
ax = plt.subplot(row[0,0])

number_of_runs = len(ordered_runs)
colors_for_action = ['#1f77b4', '#d62728','#2ca02c','#bcbd22','#ff7f0e']

for i, action_type in enumerate(action_types):

    reconstructed_proba_seq = []
    
    if i >= 2:

        break

    for s in example_states_sequence:

        reconstructed_proba_seq.append(emissionprob[s][i])

    ax.plot(np.arange(number_of_runs), reconstructed_proba_seq, label=f'Probability of {action_type}', color=colors_for_action[i], alpha=0.5, zorder=0)

ax.set_xlabel('Run index')
ax.set_ylabel('Probability of run')

ax.legend()

print(action_types)

plt.show()

