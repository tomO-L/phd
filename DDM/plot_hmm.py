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

with open(f'DDM/synthetic_data.pkl', 'rb') as file:
    synthetic_data = dill.load(file)

with open(f'DDM/best_model_aic.pkl', 'rb') as file:
    best_model = dill.load(file)

test_data = [synthetic_data[i]['choices'] for i in np.arange(10,30)]

#############
### Plots ###
#############

fig=plt.figure(figsize=(4, 7), dpi=300, constrained_layout=False, facecolor='w')
gs = fig.add_gridspec(2, 1, hspace=0.5, height_ratios=[0.3,1])
row = gs[0,0].subgridspec(1, 1)

ax1 = plt.subplot(row[0,0])

example_index = 0

example_data = np.array(test_data[0])

plot_actions_distribution(ax1, example_data)

row_bis = gs[1,0].subgridspec(2, 1, height_ratios=[0.25,1], hspace=0.1)

ax2 = plt.subplot(row_bis[0,0])
ax3 = plt.subplot(row_bis[1,0])

states = best_model.predict(np.int16(example_data.reshape(-1,1)))
plot_states_sequence(ax2, states)

plot_actions_sequence(ax3, example_data)


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


plt.show()


