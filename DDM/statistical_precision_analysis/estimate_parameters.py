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
n_simulations_list = [20]



####################
### Computations ###
####################

delta_range = np.linspace(0.01,0.1,250)

reconstructed_average_proba_sequences = compute_reconstructed_average_proba_sequences(n_simulations_list)

def callback(xk):
    print("Current solution: ", xk)

for i in range(len(n_simulations_list)):

    av_recons_proba_sequence = reconstructed_average_proba_sequences[i]

    mse_list = []

    args = args = [p_a, p_a_reward, steps_number, noise_amplitude, drift, 5000, av_recons_proba_sequence]

    # res = noisyopt.minimizeCompass(compute_mean_square_error_opt, [0.5], args=[args], paired=False, disp=True)
    res = opt.minimize(compute_mean_square_error_opt, [0.5], args=args, options={'disp': True}, callback=callback, method='Powell')

    print(res)

    # for delta in tqdm(delta_range):

    #     mse_list.append(compute_mean_square_error(delta, args))

    # min_mse = np.min(mse_list)
    # recovered_delta = delta_range[np.where(mse_list==min_mse)[0]]

    # print(recovered_delta)




quit() #################################################################################
############
### Plot ###
############

fig=plt.figure(figsize=(4, 7), dpi=300, constrained_layout=False, facecolor='w')
gs = fig.add_gridspec(1, 1, hspace=0.5,)
row = gs[0,0].subgridspec(3, 1)

ax = plt.subplot(row[:])

steps = np.arange(steps_number)

delta_range = np.linspace(0.01,0.1,10)


for delta in tqdm(delta_range):

   mean_trajectory = compute_simulations_average(p_a, p_a_reward, steps_number, noise_amplitude, delta, drift, n_simulations=300)

   ax.plot(steps, mean_trajectory, alpha=0.7)
   ax.text(steps[-1],mean_trajectory[-1], f'drift = {np.round(delta,3)}', fontsize=5)


ax.set_xlabel('Steps')
ax.set_ylabel('Average probability to chose 1')

ax.set_xticks(steps)

ax.set_ylim([0,1])

plt.show()






