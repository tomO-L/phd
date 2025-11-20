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
from hmm_functions import *

sys.excepthook = ultratb.FormattedTB(call_pdb=False)

plt.style.use('paper.mplstyle')

# Time counter
start_time = time.time()

##################
### Parameters ###
##################

n_simulations = 2
index = 9

with open(f'DDM_v2/statistical_precision_analysis/simulations_batches/n_{n_simulations}/simulations_batch_{n_simulations}_test_{index}.pkl', 'rb') as file:
    synthetic_data = dill.load(file)

with open(f'DDM_v2/statistical_precision_analysis/simulations_batches/n_{n_simulations}/best_model_score_{n_simulations}_test_{index}.pkl', 'rb') as file:
    model = dill.load(file)

test_data = [synth_data['choices'] for synth_data in synthetic_data]


####################
### Computations ###
####################

initial_state_list = []
sequences_number = len(test_data)

for i in range(sequences_number):
    
    choices_sequence = test_data[i]
    
    states_sequence = model.predict(np.int16(choices_sequence.reshape(-1,1)))
    initial_state_list.append(states_sequence[0])

initial_state_list_distri = []

for s in range(len(model.transmat_)):

    initial_state_list_distri.append(initial_state_list.count(s))

# fig, ax = plt.subplots()


transmat = model.transmat_
emission_vect = model.emissionprob_[:,1]
mat = transmat
sorted_indexes = np.argsort(emission_vect)
print(sorted_indexes)
vector = np.ones([len(transmat),1])/len(transmat)

##

new_transmat = order_matrix(mat, sorted_indexes)

##

new_mat = new_transmat

for i in range(500):

    new_mat = np.matmul(new_mat,new_transmat)

##

new_emissionmat = []
new_initial_state_list_distri = []

for i in sorted_indexes:
    new_emissionmat.append(model.emissionprob_[i,:])
    new_initial_state_list_distri.append(initial_state_list_distri[i])

new_emissionmat = np.array(new_emissionmat)
new_initial_state_list_distri = np.array(new_initial_state_list_distri)/np.sum(new_initial_state_list_distri)



delta_range = np.linspace(0.01,0.1,50)

average_proba_sequences_hmm = []

steps = np.arange(len(test_data[0]))
new_mat_i = new_transmat

for i in steps:

    new_mat_i = np.matmul(new_mat_i,new_transmat)
    res = np.matmul(new_initial_state_list_distri,new_mat_i)*new_emissionmat[:,1]
        
    average_proba_sequences_hmm.append(np.sum(res))

# test_average_probability_sequences = generate_test_average_probability_sequences(delta_range, args)

# with open(f'DDM_v2/statistical_precision_analysis/simulations_batches/test_average_probability_sequences.pkl', 'wb') as file:
#     dill.dump(test_average_probability_sequences, file)

with open(f'DDM_v2/statistical_precision_analysis/simulations_batches/test_average_probability_sequences.pkl', 'rb') as file:
    test_average_probability_sequences = dill.load(file)

mse_list = []

for test_average_probability_sequence in tqdm(test_average_probability_sequences):

    mse_list.append(compute_mean_square_error_v2(average_proba_sequences_hmm, test_average_probability_sequence))

min_mse = np.min(mse_list)
recovered_delta = delta_range[np.where(mse_list==min_mse)[0]]

with open(f'DDM_v2/statistical_precision_analysis/simulations_batches/n_{n_simulations}/recovered_delta_{n_simulations}_{index}.pkl', 'wb') as file:
    dill.dump(recovered_delta, file)


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






