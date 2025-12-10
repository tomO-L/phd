#######################
### Import packages ###
#######################

import matplotlib.pyplot as plt
import sys
import time
from IPython.core import ultratb
import dill
import numpy as np
from tqdm import tqdm
import scipy.optimize as opt
import noisyopt 
from synthetic_data_generation_functions import *
from synthetic_data_analysis_functions import *
sys.excepthook = ultratb.FormattedTB(call_pdb=False)

plt.style.use('paper.mplstyle')

# Time counter
start_time = time.time()

##################
### Parameters ###
##################

simulations_folder_path = '/home/david/Documents/code/DDM_v2_synthetic_data'
n_simulations = 20
simulations_indexes = [19] # np.arange(8,9)

with open(f'{simulations_folder_path}/n_{n_simulations}/simulations_batch_{n_simulations}_test_{5}.pkl', 'rb') as file:
    synthetic_data = dill.load(file)

test_data = [synth_data['choices'] for synth_data in synthetic_data]

with open(f'{simulations_folder_path}/n_{n_simulations}/best_model_score_{n_simulations}_test_{5}.pkl', 'rb') as file:
    model = dill.load(file)

####################
### Computations ###
####################

reconstructed_proba_sequence_list = []

for simulation_index in simulations_indexes:

    choices_sequence = test_data[simulation_index]
    steps_number = len(choices_sequence)


    reconstructed_proba_sequence = compute_reconstructed_proba_sequence(choices_sequence, model)
    reconstructed_proba_sequence_list.append(reconstructed_proba_sequence)

############
### Plot ###
############

fig=plt.figure(figsize=(4, 2), dpi=300, constrained_layout=False, facecolor='w')
gs = fig.add_gridspec(1, 1, hspace=0.5,)
row = gs[0,0].subgridspec(2, 1, height_ratios=[0.25,1])


for i, simulation_index in enumerate(simulations_indexes):

    ddm_result = synthetic_data[simulation_index]

    choice_sequence = ddm_result['choices']
    p_a_sequence = ddm_result['p_a']
    reward_sequence = ddm_result['rewards']

    steps = np.arange(len(choice_sequence))

    reconstructed_proba_sequence = reconstructed_proba_sequence_list[i]

    colors = ['violet' if i==1 else 'purple' for i in reward_sequence]

    ax1 = plt.subplot(row[0,0])
    ax1.scatter(steps, choice_sequence, label='Action Sequence', color=colors, marker='|',s=10, linewidths=2)
    ax1.set_ylabel('Run')
    ax1.set_xticks([])
    ax1.set_yticks([0,1])
    ax1.set_yticklabels(['CCW','CW'])

    ax2 = plt.subplot(row[1,0])
    
    ax2.axhline(0.5, linestyle='--', color='grey', zorder=0)
    ax2.plot(steps, p_a_sequence, label='Probability Sequence', color='k', marker='+', markersize=3, linewidth=0., markeredgewidth=0.8)
    # ax2.plot(steps, p_a_sequence, label='Probability Sequence', alpha=0.2)
    # ax2.scatter(steps, p_a_sequence, label='Probability Sequence', alpha=0.2, marker='+')
    ax2.plot(steps, reconstructed_proba_sequence, label='Reconstructed Probability Sequence', color='green', marker='+', markersize=3, linewidth=0., alpha=0.5, markeredgewidth=0.8)

    ax2.set_xlabel('Step')
    ax2.set_ylabel('Probability\nto do CW')
    ax2.set_xticks(steps[::10])
    ax2.set_ylim([-0.05,1.05])
    ax2.legend()

plt.show()






