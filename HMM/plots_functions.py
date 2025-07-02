#######################
### Import packages ###
#######################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import os
import pickle
from hmmlearn import hmm, vhmm
from tqdm import tqdm

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

def plot_states_sequence(ax, states, xticks=[], show_cbar = True):

    cmap = plt.cm.plasma
    norm = Normalize(vmin=np.nanmin(states), vmax=np.nanmax(states))

    # for i in set(states):

    #     print(i)
    #     print(colors[i])

    
    ax.imshow(states, cmap=cmap, aspect='auto')

    # ax.set_xlim([0,len(states)+1])
    # ax.set_ylim([-3,3])

    ax.set_xticks(xticks)
    ax.set_yticks([])

    ticks = np.arange(np.nanmax(states)+1)

    fig = ax.get_figure()

    if show_cbar:
        cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, drawedges=False, orientation='horizontal', label='States', location='top', ticks=ticks)

    



