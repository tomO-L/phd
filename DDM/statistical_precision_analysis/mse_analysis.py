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


fig=plt.figure(figsize=(4, 7), dpi=300, constrained_layout=False, facecolor='w')
gs = fig.add_gridspec(1, 1)
row = gs[0,0].subgridspec(2, 2, hspace=0.5)

###################################################################

ax1 = plt.subplot(row[0,:])

# n_simulations_list = [20,60,100,200,600]
# n_simulations_list = [20,60,100,200,600,1000,2000,3000,5000]
n_simulations_list = [20,60,100,200,600,1000,2000,5000]
example_n_simulations = 5000

steps_number = 20
noise_amplitude = 0.1
# delta = 0.05
drift = 0.0
p_a = 0.5
p_a_reward = 1

with open(f'DDM/statistical_precision_analysis/simulations_batches/simulations_batch_{example_n_simulations}.pkl', 'rb') as file:
    synthetic_data = dill.load(file)

test_data = [synth_data['choices'] for synth_data in synthetic_data]

with open(f'DDM/statistical_precision_analysis/simulations_batches/best_model_score_{example_n_simulations}_fulltraining.pkl', 'rb') as file:
    model = dill.load(file)

states_sequences = []
sequences_number = len(test_data)

for i in range(sequences_number):
    
    choices_sequence = test_data[i]
    
    states_sequence = model.predict(np.int16(choices_sequence.reshape(-1,1)))
    states_sequences.append(states_sequence)

emissionprob = model.emissionprob_


reconstructed_p_a_sequences = []

for i in range(len(states_sequences)):

    reconstructed_p_a_sequence = []

    for s in states_sequences[i]:

        reconstructed_p_a_sequence.append(emissionprob[s][1])

    reconstructed_p_a_sequences.append(reconstructed_p_a_sequence)

reconstructed_average_p_a = np.mean(reconstructed_p_a_sequences,axis=0)


x = np.arange(steps_number)

delta_range = [0.03,0.04,0.05,0.06,0.07] #np.linspace(0.01,0.1,10)

for delta in tqdm(delta_range):

    mean_probability = compute_simulations_average(p_a, p_a_reward, steps_number, noise_amplitude, delta, drift, n_simulations=50)

    ax1.plot(x, mean_probability, alpha=0.5, linestyle='--')
    ax1.text(x[-1],mean_probability[-1], f'drift = {np.round(delta,3)}', fontsize=5)

ax1.plot(x,reconstructed_average_p_a, color='k', label='Reconstructed average probability')

ax1.set_xticks(x)
ax1.set_xlabel('Steps')
ax1.set_ylabel('Probability to chose CW')

ax1.legend()

###################################################################

# ax2 = plt.subplot(row[1,0])
ax3 = plt.subplot(row[1,:])

recovered_delta_list = []

for n_simulations in n_simulations_list:

    with open(f'DDM/statistical_precision_analysis/simulations_batches/mse_{n_simulations}_fulltraining2_auto.pkl', 'rb') as file: ### NEED MSE
        delta_range,mse_list = dill.load(file)
##############################################################
##############################################################
##############################################################
##############################################################

    min_mse = np.min(mse_list)
    
    recovered_delta = delta_range[np.where(mse_list==min_mse)[0]]
    print(recovered_delta)
    recovered_delta_list.append(recovered_delta)

    # ax2.plot(delta_range,mse_list, label=f"{n_simulations} simulations", alpha=0.5)
    # ax2.scatter(recovered_delta,min_mse, marker='+', c='k')

# ax2.axvline(0.05, linewidth=0.7, color='k', linestyle='--', label='Drift to recover')


ax3.scatter(n_simulations_list, recovered_delta_list, marker='+')

ax3.axhline(0.05, linewidth=0.7, color='k', linestyle='--', label='Drift to recover')
# ax.axvline(recovered_delta, linewidth=0.7,color='grey', linestyle='--', label='Recovered Drift (i.e with minimum MSE)')

# ax2.set_xlabel('Drift')
# ax2.set_ylabel('Mean square error')

ax3.set_xscale('log')

ax3.set_xlabel('Nbr. of simulations')
ax3.set_ylabel('Recovered drift')

# ax2.legend(ncols=5, loc=(0,1))
ax3.legend()

plt.show()
