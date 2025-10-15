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

steps_number = 20
noise_amplitude = 0.1
# delta = 0.05
drift = 0.0
p_a = 0.5
p_a_reward = 1

# np.random.seed(58777) # initial seed
# np.random.seed(587) # test seed
# np.random.seed(50) # test seed

# n_simulations_list = [20,60,100,200,600,1000,2000]
n_simulations_list = [600]



####################
### Computations ###
####################

delta_range = np.linspace(0.01,0.1,250)

# reconstructed_average_proba_sequences = compute_reconstructed_average_proba_sequences(n_simulations_list)
reconstructed_average_proba_sequences = compute_reconstructed_average_proba_sequences_fulltraining(n_simulations_list)

def callback(xk):
    print("Current solution: ", xk)

recompute_mse = True

for i in range(len(n_simulations_list)):

    if not(recompute_mse):

        with open(f'DDM/statistical_precision_analysis/simulations_batches/mse_{n_simulations_list[i]}_fulltraining.pkl', 'rb') as file:
            delta_range,mse_list = dill.load([delta_range,mse_list], file)

        continue

    av_recons_proba_sequence = reconstructed_average_proba_sequences[i]

    mse_list = []

    args = args = [p_a, p_a_reward, steps_number, noise_amplitude, drift, 5000, av_recons_proba_sequence]

    # res = noisyopt.minimizeCompass(compute_mean_square_error_opt, [0.5], args=[args], paired=False, disp=True)
    # res = opt.minimize(compute_mean_square_error_opt, [0.5], args=args, options={'disp': True}, callback=callback, method='Powell')

    # print(res)

    for delta in tqdm(delta_range):

        mse_list.append(compute_mean_square_error(delta, args))

    min_mse = np.min(mse_list)
    recovered_delta = delta_range[np.where(mse_list==min_mse)[0]]

    print(recovered_delta)

    with open(f'DDM/statistical_precision_analysis/simulations_batches/mse_{n_simulations_list[i]}_fulltraining.pkl', 'wb') as file:
        dill.dump([delta_range,mse_list], file)

# quit() #################################################################################
############
### Plot ###
############

fig=plt.figure(figsize=(4, 7), dpi=300, constrained_layout=False, facecolor='w')
gs = fig.add_gridspec(1, 1, hspace=0.5,)
row = gs[0,0].subgridspec(3, 1)

ax = plt.subplot(row[:])

ax.plot(delta_range,mse_list, label="Mean Square Error for different Drift values")
ax.axvline(0.05, linewidth=0.7, color='k', linestyle='--', label='Drift used to generate simulations')
ax.axvline(recovered_delta, linewidth=0.7,color='grey', linestyle='--', label='Recovered Drift (i.e with minimum MSE)')

ax.set_xlabel('Delta')
ax.set_ylabel('Mean Square Error')

ax.legend()

plt.show()






