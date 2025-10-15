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
gs = fig.add_gridspec(1, 1, hspace=0.5,)
row = gs[0,0].subgridspec(3, 1)

ax = plt.subplot(row[:])

# n_simulations_list = [20,60,100,200,600,1000,2000]
n_simulations_list = [20,60,100,200,600]

for n_simulations in n_simulations_list:

    with open(f'DDM/statistical_precision_analysis/simulations_batches/mse_{n_simulations}_fulltraining.pkl', 'rb') as file:
        delta_range,mse_list = dill.load(file)

    min_mse = np.min(mse_list)
    recovered_delta = delta_range[np.where(mse_list==min_mse)[0]]

    ax.plot(delta_range,mse_list, label=f"{n_simulations} simulations", alpha=0.5)
    ax.scatter(recovered_delta,min_mse, marker='+', c='k')

ax.axvline(0.05, linewidth=0.7, color='k', linestyle='--', label='Drift used to generate simulations')
# ax.axvline(recovered_delta, linewidth=0.7,color='grey', linestyle='--', label='Recovered Drift (i.e with minimum MSE)')

ax.set_xlabel('Delta')
ax.set_ylabel('Mean Square Error')

ax.legend()

plt.show()
