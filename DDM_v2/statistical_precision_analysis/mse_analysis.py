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

sample_size = 9
n_simulations = 60

#############
### Plots ###
#############

fig=plt.figure(figsize=(1, 4), dpi=300, constrained_layout=False, facecolor='w')
gs = fig.add_gridspec(1, 1)
row = gs[:].subgridspec(1, 1, hspace=0.5)

ax1 = plt.subplot(row[0,0])

recovered_delta_list = []

for index in range(1,sample_size+1):

    with open(f'DDM_v2/statistical_precision_analysis/simulations_batches/recovered_delta_{n_simulations}_{index}.pkl', 'rb') as file:
        recovered_delta = dill.load(file)
    
    recovered_delta_list.append(recovered_delta)

ax1.scatter(np.ones(sample_size), recovered_delta_list, marker='+', alpha=0.5)

ax1.axhline(0.05, linewidth=0.7, color='k', linestyle='--', label='Drift to recover')

ax1.set_xticks([])
ax1.set_ylabel('Recovered drift')

# ax2.legend(ncols=5, loc=(0,1))
# ax3.legend()

plt.show()
