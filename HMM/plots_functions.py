#######################
### Import packages ###
#######################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from scipy.stats import wilcoxon
import os
import pickle
from hmmlearn import hmm, vhmm
from tqdm import tqdm
from analysis_functions import *

########################
### Define functions ###
########################


def plot_actions_distribution(ax, ordered_actions_types_number,
                            action_types = ['run_around_tower', 'run_between_towers', 'run_toward_tower', 'exploratory_run']):

    gen_action_types_ditribution = [np.count_nonzero(ordered_actions_types_number==i) for i in range(len(action_types))]
                                
    ax.bar(action_types, gen_action_types_ditribution)



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

        ax.broken_barh(x_barh, y_barh)

    if show_yticks:
        ax.set_yticks(np.arange(len(action_types)), action_types)

# def plot_states_sequence(ax, states):

#     cmap = plt.cm.viridis
#     norm = Normalize(vmin=0, vmax=max(states))

#     colors = cmap(norm(states))

#     # for i in set(states):

#     #     print(i)
#     #     print(colors[i])

#     for i,s in enumerate(states):

#         ax.add_patch(Rectangle((i, 0), 1, 2,color=colors[s]))
#         # ax.axvspan(i,i+1,color=colors[s])

#     ax.set_xlabel('Rank')

#     ax.set_xlim([0,len(states)+1])
#     ax.set_ylim([-5,5])

#     fig = ax.get_figure()
#     cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, drawedges=False)

def plot_states_sequence(ax, states, xticks=[], colors= ['#d62728','#7f7f7f','#bcbd22','#2ca02c','#1f77b4','#ff7f0e'], show_cbar = True):

    cmap = ListedColormap(colors) #plt.cm.Set1
    norm = Normalize(vmin=np.nanmin(states), vmax=np.nanmax(states)+1)

    # for i in set(states):

    #     print(i)
    #     print(colors[i])

    
    ax.imshow(states, cmap=cmap, aspect='auto', interpolation='none')

    ax.set_xlabel('Rank')
    ax.set_ylabel('Sessions')

    # ax.set_xlim([0,len(states)+1])
    # ax.set_ylim([-3,3])

    ax.set_xticks(xticks)
    ax.set_yticks([])

    ticks = np.arange(np.nanmax(states)+1)

    fig = ax.get_figure()

    if show_cbar:
        cbar = fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap), ax=ax, drawedges=False, orientation='horizontal', label='States', location='top')
        cbar.set_ticks(ticks=ticks+0.5, labels=np.int8(ticks))

    
def plot_states_distribution(states_sequence,ax):

    states_id = list(set(states_sequence))

    states_distribution = np.array([np.count_nonzero(states_sequence==i) for i in states_id])/len(states_sequence)

    ax.bar(states_id, states_distribution)
    
    ax.set_xlabel('States')
    ax.set_ylabel('% of step in each state')

def one_sample_wilcoxon_test(sample, med0):

    w, p = wilcoxon(np.array(sample) - med0)

    return p

def plot_learning_curve(values_persessions_permouse, mice_to_analyse, ax, wilcoxon_test_h0=np.nan):

    all_mice_values_persessions = []

    for mouse in mice_to_analyse:

        values_persessions = values_persessions_permouse[mouse]

        values_persessions = np.transpose(values_persessions)
        all_mice_values_persessions.append(values_persessions[1])

        # Plot individual mice
        ax.plot(values_persessions[0],values_persessions[1], alpha=0.3)

    # Plot median and quartiles
    median_values = np.nanmedian(all_mice_values_persessions, axis=0)
    upper_quartile_values = np.nanpercentile(all_mice_values_persessions, 75, axis=0)
    lower_quartile_values = np.nanpercentile(all_mice_values_persessions, 25, axis=0)

    # Perform one sample Wilcoxon test

    if not(np.isnan(wilcoxon_test_h0)):

        for i in range(len(values_persessions[0])):

            p = one_sample_wilcoxon_test(np.transpose(all_mice_values_persessions)[i], wilcoxon_test_h0)

            # print(p)
        
            if p<=0.05:

                ax.scatter(i+1,median_values[i],color='#1f77b4', edgecolor='black', marker='*',s=20, linewidths=0.5, zorder=1000)

    ax.errorbar(values_persessions[0],median_values, yerr=[median_values-lower_quartile_values, upper_quartile_values-median_values], color='black')
    
    ax.set_xticks(values_persessions[0])


def plot_states_distri_across_sessions(states_distributions, ax, colors = ['#d62728','#7f7f7f','#bcbd22','#2ca02c','#1f77b4','#ff7f0e']):

    sessions_index = np.arange(len(states_distributions[0]))+1

    for i in range(len(states_distributions)):

        ax.plot(sessions_index,states_distributions[i], color=colors[i], label=f'state {i}', marker='+')

    ax.set_xlabel('Session')
    ax.set_xticks(sessions_index)
    
    ax.set_ylabel('States ratio')

    ax.legend()
    # ax.set_ylim([0,1])

def plot_cumulated_turns_profile(ordered_runs, ax, states_sequence = None, colors = ['#d62728','#7f7f7f','#bcbd22','#2ca02c','#1f77b4','#ff7f0e']):

    rank_of_rewarded_qts, rank_of_unrewarded_qts = compute_cumulated_turns_profile(ordered_runs)

    cumulated_rewarded_qts = np.arange(len(rank_of_rewarded_qts))+1
    cumulated_unrewarded_qts = np.arange(len(rank_of_unrewarded_qts))+1

    ax.step(rank_of_rewarded_qts, cumulated_rewarded_qts, where='post', label='Rewarded direction', color='black')
    ax.step(rank_of_unrewarded_qts, cumulated_unrewarded_qts, where='post', label='Unrewarded direction', color='black', linestyle='--')

    for i in range(len(ordered_runs)):

        ax.axvspan(i-0.5,i+0.5, color=colors[states_sequence[i]], zorder=0)


    cmap = ListedColormap(colors) #plt.cm.Set1
    norm = Normalize(vmin=np.nanmin(states_sequence), vmax=np.nanmax(states_sequence)+1)

    ticks = np.arange(np.nanmax(states_sequence)+1)

    fig = ax.get_figure()

    cbar = fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap), ax=ax, drawedges=False, orientation='horizontal', label='States', location='top')
    cbar.set_ticks(ticks=ticks+0.5, labels=np.int8(ticks))

def plot_cumulated_states_profile(states_sequence, states, ax, colors = ['#d62728','#7f7f7f','#bcbd22','#2ca02c','#1f77b4','#ff7f0e']):

    cumulated_states_profile = compute_cumulated_states_profile(states_sequence, states)

    for s in states:

        ax.step(cumulated_states_profile[s], np.arange(len(cumulated_states_profile[s]))+1, color=colors[s])

    ax.set_xlabel('Rank')
    ax.set_ylabel('Cumulated step in state')

def plot_occurence_frequency(states_sequence, states_type, ax, colors = ['#d62728','#7f7f7f','#bcbd22','#2ca02c','#1f77b4','#ff7f0e'], window_size=50, weight=1):

    length = len(states_sequence)

    for s in states_type:

        sequence = np.where(states_sequence==s,1,0)

        occurence_frequency = compute_occurence_frequency_v1(sequence, window_size)

        ax.plot(np.arange(length), occurence_frequency, color=colors[s])

    cmap = ListedColormap(colors) #plt.cm.Set1
    norm = Normalize(vmin=np.nanmin(states_sequence), vmax=np.nanmax(states_sequence)+1)

    ticks = np.arange(np.nanmax(states_sequence)+1)

    fig = ax.get_figure()

    cbar = fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap), ax=ax, drawedges=False, orientation='horizontal', label='States', location='top')
    cbar.set_ticks(ticks=ticks+0.5, labels=np.int8(ticks))

    ax.set_xlabel('Rank')
    ax.set_ylabel(f'Occurence frequency\n in a window of size {window_size}')


    
        
