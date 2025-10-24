#######################
### Import packages ###
#######################

import matplotlib.pyplot as plt
import sys
import time
from IPython.core import ultratb
import dill
import numpy as np
from synthetic_data_generation_functions import *
from tqdm import tqdm
sys.excepthook = ultratb.FormattedTB(call_pdb=False)

plt.style.use('paper.mplstyle')

# Time counter
start_time = time.time()

##################
### Parameters ###
##################

steps_number = 20
noise_amplitude = 0.1
delta = 0.05
drift = 0.0
p_a = 0.5
p_a_reward = 1
n_simulations_list = [20,60,100,200,600,1000,2000, 3000, 5000]
# n_simulations_list = [3000]

for n_simulations in n_simulations_list:
    
    simulations_batch = run_simulations_batch(p_a, p_a_reward, steps_number, noise_amplitude, delta, drift, n_simulations)

    with open(f'DDM/statistical_precision_analysis/simulations_batches/simulations_batch_{n_simulations}_test2.pkl', 'wb') as file:
        dill.dump(simulations_batch, file)



