#######################
### Import packages ###
#######################

from plots_functions import *
from hmm_functions import *
import matplotlib.pyplot as plt
from hmmlearn import hmm, vhmm
import sys
import time
from IPython.core import ultratb
import dill
sys.excepthook = ultratb.FormattedTB(call_pdb=False)

plt.style.use('paper.mplstyle')

###################
### Import Data ###
###################

# with open(f'DDM/synthetic_data_test.pkl', 'rb') as file:
#     synthetic_data = dill.load(file)

# with open(f'DDM/best_model_score_2-10.pkl', 'rb') as file:
#     model = dill.load(file)

# with open(f'DDM/simple_synthetic_data_test.pkl', 'rb') as file:
#     synthetic_data = dill.load(file)

with open(f'DDM/simple_synthetic_data_test.pkl', 'rb') as file:
    synthetic_data = dill.load(file)

with open(f'DDM/simple_best_model_score_2-40.pkl', 'rb') as file:
    model = dill.load(file)


# test_data = [synthetic_data[i]['choices'] for i in np.arange(200,300)]
test_data = [synth_data['choices'] for synth_data in synthetic_data]

###################
### Using model ###
###################

states_sequences = []
sequences_number = len(test_data)

for i in range(sequences_number):
    
    choices_sequence = test_data[i]
    choices_number = len(test_data[i])
    
    states_sequence = model.predict(np.int16(choices_sequence.reshape(-1,1)))
    states_sequences.append(states_sequence)

# states_distributions = compute_states_distribution_persession(path_to_data_folder,mouse_name,sessions_index,model)
# states_time_distributions, states_time_ratio_distributions = compute_time_in_states_persession(path_to_data_folder,mouse_name,sessions_index,model)

#############
### Plots ###
#############

### States Sequences ###

max_length = 0
# colors = ['red','blue']
# colors = plt.cm.viridis
# norm = Normalize(vmin=0, vmax=len(model.transmat_))
# colors = cmap(norm(local_index))

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
    
plot_states_sequences(ax, padded_states_sequences)

ax.set_xticks(np.arange(0,max_length,10))
# ax.set_yticks(np.arange(0,sequences_number,100))

### Action and transition matrixes ###

fig=plt.figure(figsize=(3.5, 3), dpi=300, constrained_layout=False, facecolor='w')
gs = fig.add_gridspec(1, 1)
row = gs[0].subgridspec(1,2)
ax5 = plt.subplot(row[0,0])
ax6 = plt.subplot(row[0,1])

states_number = len(model.transmat_)

ax5.imshow(model.transmat_)
ax5.set_xticks(np.arange(states_number))
ax5.set_yticks(np.arange(states_number))

ax5.set_title('Transition matrix')
ax5.set_xlabel('To state')
ax5.set_ylabel('From state')


ax6.imshow(model.emissionprob_)
ax6.set_xticks([0,1], labels=[0,1], rotation=30, ha="right", rotation_mode="anchor")
ax6.set_yticks(np.arange(states_number))
ax6.set_title('Emission matrix')
ax6.set_xlabel('Choice')
ax6.set_ylabel('State')


### Test Plots ###

example_run_index = 4850
# example_run = synthetic_data[200:][example_run_index]
example_run = synthetic_data[example_run_index]

cmap = plt.cm.viridis # ListedColormap(colors) #plt.cm.Set1
norm = Normalize(vmin=0, vmax=len(model.transmat_)-1)
colormap = cm.ScalarMappable(norm=norm,cmap=cmap)
color_for_states = ['k']*len(states_sequences[example_run_index]) #[colormap.to_rgba([i])[0] for i in states_sequences[example_run_index]]

fig=plt.figure(figsize=(4, 7), dpi=300, constrained_layout=False, facecolor='w')
gs = fig.add_gridspec(1, 1, hspace=0.5,)
row = gs[0,0].subgridspec(3, 1)

    
reward_sequence = example_run['rewards']
choice_sequence = example_run['choices']
p_a_sequence = example_run['p_a']
drift_sequence = example_run['drift']

reconstructed_p_a_sequence = []

emissionprob = model.emissionprob_

for s in states_sequences[example_run_index]:

    reconstructed_p_a_sequence.append(emissionprob[s][1])

steps_number = len(choice_sequence)
steps = np.arange(steps_number)

ax1 = plt.subplot(row[0,0])
ax1.scatter(steps, reward_sequence, label='Reward Sequence', marker='+', c=color_for_states)
ax1.set_ylabel('Reward')
ax1.set_xticks([])
ax1.set_yticks([0,1])
ax1.set_title(f'Simulation {example_run_index}')

ax2 = plt.subplot(row[1,0])
ax2.scatter(steps, choice_sequence, label='Choice Sequence', marker='+', c=color_for_states)
ax2.set_ylabel('Choice')
ax2.set_xticks([])
ax2.set_yticks([0,1])

ax3 = plt.subplot(row[2,0])
ax3.plot(steps, p_a_sequence, label='Probability Sequence of 1', color='k', alpha=0.5, zorder=0)
ax3.scatter(steps, p_a_sequence, marker='+', c=color_for_states)
# ax3.plot(steps, reconstructed_p_a_sequence, color='red', alpha=0.5, zorder=0)

ax3.set_ylabel('Probability\nto chose 1')
ax3.set_xticks(steps)
ax3.set_ylim([-0.05,1.05])

# ax4 = plt.subplot(row[3,0])
# ax4.plot(steps, drift_sequence, label='Drift Coefficient', color='k', alpha=0.5, zorder=0)
# ax4.scatter(steps, drift_sequence, marker='+', c=color_for_states)
# ax4.set_ylabel('Drift Coefficient')
# ax4.set_xticks(steps)
# ax4.set_ylim([-0.05,None])


### Residuals Histogram ###

fig=plt.figure(figsize=(4, 4), dpi=300, constrained_layout=False, facecolor='w')
gs = fig.add_gridspec(1, 1, hspace=0.5,)
row = gs[0,0].subgridspec(1, 1)

ax = plt.subplot(row[0,0])

def plot_residuals_hist(p_a_sequences, reconstructed_p_a_sequences, ax, bins=None):

    residuals_list = []
    
    for i in range(len(p_a_sequences)):

        residuals = np.sum((p_a_sequences[i] - reconstructed_p_a_sequences[i])**2)/len(reconstructed_p_a_sequences[i])
        residuals_list.append(residuals)

    ax.hist(residuals_list, bins=bins)

    ax.set_xlabel('Residuals of true and infered P(1)')
    ax.set_ylabel('Number of simulations')

    return residuals_list

emissionprob = model.emissionprob_

reconstructed_p_a_sequences = []

for states_seq in states_sequences:

    reconstructed_p_a_seq = []

    for s in states_seq:

        reconstructed_p_a_seq.append(emissionprob[s][1])

    reconstructed_p_a_sequences.append(reconstructed_p_a_seq)

p_a_sequences = [synth_data['p_a'] for synth_data in synthetic_data]

residuals_list = plot_residuals_hist(p_a_sequences, reconstructed_p_a_sequences, ax, bins=40)

# with open(f'DDM/residuals_aic_2-10.pkl', 'wb') as file:
#     dill.dump(residuals_list, file)

#residuals = np.sum((p_a_sequence - reconstructed_p_a_sequence)**2)/len(reconstructed_p_a_sequence)
#print(f"### Residuals: r = {residuals} ###")

### Time to reach solution ###

fig=plt.figure(figsize=(3.5, 3), dpi=300, constrained_layout=False, facecolor='w')
gs = fig.add_gridspec(1, 1)
row = gs[0].subgridspec(1,1)
ax = plt.subplot(row[:])

def find_final_state_start(states_sequence,success_state):

    success_step = 0

    for i in range(len(states_sequence)):

        if states_sequence[i]==success_state:

            break

        success_step += 1

    return success_step

def find_threshold_cross(p_a_sequence,threshold):

    success_step = 0

    for i in range(len(p_a_sequence)):

        if p_a_sequence[i]>=threshold:

            break

        success_step += 1

    return success_step


final_state_start_list = []
threshold_cross_list = []

for i,s_seq in enumerate(states_sequences):

    threshold_cross_list.append(find_threshold_cross(synthetic_data[i]['p_a'], 0.8))
    final_state_start_list.append(find_final_state_start(s_seq,1))

ax.hist(threshold_cross_list, bins=np.arange(20), align='left', alpha=0.5, label='First step at P(1)=1')
ax.hist(final_state_start_list, bins=np.arange(20), align='left', alpha=0.5, label='First step in final state')

ax.set_xlabel("Step")
ax.set_ylabel("Number of Simulations")
ax.set_xticks(np.arange(len(states_sequences[0])))
ax.legend()

### Mean Trajectory ###

fig=plt.figure(figsize=(3.5, 3), dpi=300, constrained_layout=False, facecolor='w')
gs = fig.add_gridspec(1, 1)
row = gs[0].subgridspec(1,1)
ax = plt.subplot(row[:])

mean_trajectory = np.mean(p_a_sequences,axis=0)
mean_trajectory_std = np.std(p_a_sequences,axis=0)

reconstructed_mean_trajectory = np.mean(reconstructed_p_a_sequences,axis=0)
reconstructed_mean_trajectory_std = np.std(reconstructed_p_a_sequences,axis=0)

# median_trajectory = np.median(p_a_sequences,axis=0)
# trajectory_upper_quartile = np.percentile(p_a_sequences,75,axis=0)
# trajectory_lower_quartile = np.percentile(p_a_sequences,25,axis=0)
# bars = [median_trajectory-trajectory_lower_quartile,trajectory_upper_quartile-median_trajectory]

reconstructed_mean_trajectory = np.mean(reconstructed_p_a_sequences,axis=0)
reconstructed_mean_trajectory_std = np.std(reconstructed_p_a_sequences,axis=0)

# median_reconstructed_trajectory = np.median(reconstructed_p_a_sequences,axis=0)
# reconstructed_trajectory_upper_quartile = np.percentile(reconstructed_p_a_sequences,75,axis=0)
# reconstructed_trajectory_lower_quartile = np.percentile(reconstructed_p_a_sequences,25,axis=0)
# reconstructed_bars = [median_reconstructed_trajectory-reconstructed_trajectory_lower_quartile,reconstructed_trajectory_upper_quartile-median_reconstructed_trajectory]

print(mean_trajectory_std)
print(reconstructed_mean_trajectory_std)
# print(median_reconstructed_trajectory)

# ax.errorbar(steps, mean_trajectory, yerr=mean_trajectory_std, alpha=0.5)
# ax.errorbar(steps, reconstructed_mean_trajectory, yerr=reconstructed_mean_trajectory_std, alpha=0.5)

ax.plot(steps, mean_trajectory, alpha=0.5, marker='+')
ax.plot(steps, reconstructed_mean_trajectory, alpha=0.5, marker='+')

# ax.errorbar(steps, median_trajectory, yerr=bars, alpha=0.5)
# ax.errorbar(steps, median_reconstructed_trajectory, yerr=reconstructed_bars, alpha=0.5)

ax.set_ylim([0,1])
ax.set_xlabel("Step")
ax.set_ylabel("Probability to chose 1")
ax.set_xticks(np.arange(len(states_sequences[0])))
ax.legend()

### Probability Distribution per step ###

fig=plt.figure(figsize=(9, 1), dpi=300, constrained_layout=False, facecolor='w')
gs = fig.add_gridspec(1, 1)
row = gs[0].subgridspec(2,steps_number)

for i in range(steps_number):

    ax = plt.subplot(row[0,i])

    values = np.array(p_a_sequences)[:,i]
    reconstructed_values = np.array(reconstructed_p_a_sequences)[:,i]
    
    # print(reconstructed_values)

    ax.hist(values, bins=np.arange(11)*0.1, alpha=0.5)
    ax.hist(reconstructed_values, bins=np.arange(11)*0.1, alpha=0.5)

    ax.set_ylim([0,5000])

    if i!=0:

        ax.set_xticks([])
        ax.set_yticks([])

# ax = plt.subplot(row[1,:])

# for reconstructed_p_a_seq in reconstructed_p_a_sequences:

#     ax.plot(steps,reconstructed_p_a_seq, alpha=0.05, color='grey')
#     ax.set_xticks(steps)

#ax.set_ylim([0,1])
#ax.set_xlabel("Step")
#ax.set_ylabel("Probability to chose 1")
#ax.set_xticks(np.arange(len(states_sequences[0])))
ax.legend()

# all_epochs = load_pickle_data(folder_path_mouse_to_analyse, example_session_index)["all_epochs"]
# ordered_runs = order_runs(all_epochs)[0]

# example_states_sequence = states_sequences[example_session_index]

# window_size = 50

# ## 
# fig=plt.figure(figsize=(7, 4), dpi=300, constrained_layout=False, facecolor='w')
# gs = fig.add_gridspec(1, 1)
# row = gs[0,0].subgridspec(1, 1)
# ax = plt.subplot(row[0,0])

# plot_cumulated_turns_profile(ordered_runs, ax, states_sequence = example_states_sequence, colors = colors)

# ## 
# fig=plt.figure(figsize=(7, 4), dpi=300, constrained_layout=False, facecolor='w')
# gs = fig.add_gridspec(1, 1)
# row = gs[0,0].subgridspec(1, 1)
# ax = plt.subplot(row[0,0])

# plot_cumulated_states_profile(example_states_sequence, np.arange(states_number), ax, colors = colors)

# ## 
# fig=plt.figure(figsize=(7, 4), dpi=300, constrained_layout=False, facecolor='w')
# gs = fig.add_gridspec(1, 1)
# row = gs[0,0].subgridspec(1, 1)
# ax = plt.subplot(row[0,0])

# plot_reward_rate(ordered_runs, ax, window_size=window_size)
# plot_states_occurence_frequency(example_states_sequence, np.arange(states_number), ax, colors = colors, window_size=window_size)

plt.show()