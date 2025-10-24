#######################
### Import packages ###
#######################

from hmm_functions import *
import matplotlib.pyplot as plt
from hmmlearn import hmm, vhmm
import sys
import time
from IPython.core import ultratb
import dill
import numpy as np
sys.excepthook = ultratb.FormattedTB(call_pdb=False)

plt.style.use('paper.mplstyle')

start_time = time.time()

n_simulations = 5000

with open(f'DDM/statistical_precision_analysis/simulations_batches/simulations_batch_{n_simulations}_test2.pkl', 'rb') as file:
    synthetic_data = dill.load(file)

# slice_size = int(n_simulations/4)
slice_size = int(n_simulations/2)

training_data = [synthetic_data[i]['choices'] for i in np.arange(0,slice_size)]
validation_data = [synthetic_data[i]['choices'] for i in np.arange(slice_size,2*slice_size)]



###################
### Infer model ###
###################

## Reformating data

training_emissions = np.array([]).astype(int)
validation_emissions = np.array([]).astype(int)

for x,y in zip(training_data,validation_data):

    training_emissions = np.concatenate((training_emissions, x))
    validation_emissions = np.concatenate((validation_emissions, y))

training_emissions = training_emissions.reshape(-1,1)
training_emissions_lengths = [len(x) for x in training_data]

validation_emissions = validation_emissions.reshape(-1,1)
validation_emissions_lengths = [len(y) for y in validation_data]

## Infer best model

# n_to_test = np.arange(2,10)
n_to_test = np.arange(2,15)


best_model, best_score = infer_best_model_score(training_emissions, validation_emissions, 
                                          training_emissions_lengths, validation_emissions_lengths, 
                                          n_to_test, seed=13)

###########################
### Save model and sets ###
###########################

# with open(f'DDM/statistical_precision_analysis/simulations_batches/best_model_score_{n_simulations}.pkl', 'wb') as file:
#     dill.dump(best_model, file)

with open(f'DDM/statistical_precision_analysis/simulations_batches/best_model_score_{n_simulations}_fulltraining_2.pkl', 'wb') as file:
    dill.dump(best_model, file)

end_time = time.time()

print(f"It took {(end_time-start_time)//60} min {(end_time-start_time)%60} s")


