#######################
### Import packages ###
#######################

import numpy as np
import matplotlib.pyplot as plt
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






