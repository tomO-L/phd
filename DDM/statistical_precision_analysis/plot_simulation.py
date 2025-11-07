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

n_simulations = 5000
simulation_index = 5
with open(f'DDM/statistical_precision_analysis/simulations_batches/simulations_batch_{n_simulations}_test.pkl', 'rb') as file:
    synthetic_data = dill.load(file)

test_data = [synth_data['choices'] for synth_data in synthetic_data]

with open(f'DDM/statistical_precision_analysis/simulations_batches/best_model_score_{n_simulations}_fulltraining_2.pkl', 'rb') as file:
    model = dill.load(file)

choices_sequence = test_data[simulation_index]
steps_number = len(choices_sequence)

####################
### Computations ###
####################

reconstructed_proba_sequence = compute_reconstructed_proba_sequence(choices_sequence, model)

############
### Plot ###
############

fig=plt.figure(figsize=(4, 2), dpi=300, constrained_layout=False, facecolor='w')
gs = fig.add_gridspec(1, 1, hspace=0.5,)
row = gs[0,0].subgridspec(2, 1)

steps = np.arange(steps_number)

ddm_result = synthetic_data[simulation_index]

# reward_sequence = ddm_result['rewards']
choice_sequence = ddm_result['choices']
p_a_sequence = ddm_result['p_a']

# ax1 = plt.subplot(row[0,0])
# ax1.plot(steps, reward_sequence, label='Reward Sequence', color='k')
# ax1.set_ylabel('Reward')
# ax1.set_xticks([])
# ax1.set_yticks([0,1])

ax1 = plt.subplot(row[0,0])
ax1.scatter(steps, choice_sequence, label='Action Sequence', color='k', marker='+')
ax1.set_ylabel('Action')
ax1.set_xticks([])
ax1.set_yticks([0,1])
ax1.set_yticklabels(['B','A'])

ax2 = plt.subplot(row[1,0])
ax2.scatter(steps, p_a_sequence, label='Probability Sequence', color='blue', alpha=0.5, marker='+')
# ax2.scatter(steps, reconstructed_proba_sequence, label='Reconstructed Probability Sequence', color='green', marker='+', alpha=0.5)

ax2.set_xlabel('Step')
ax2.set_ylabel('Probability\nto do A')
ax2.set_xticks(steps)
ax2.set_ylim([-0.05,1.05])
ax2.legend()

plt.show()






