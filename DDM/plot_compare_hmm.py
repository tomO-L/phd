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

##############################
### Import Data and Models ###
##############################

with open(f'DDM/residuals_aic_2-10.pkl', 'rb') as file:
    residuals_aic_2to10 = dill.load(file)

with open(f'DDM/residuals_score_2-10.pkl', 'rb') as file:
    residuals_score_2to10 = dill.load(file)

# with open(f'DDM/residuals_score_2-20.pkl', 'rb') as file:
#     residuals_score_2to20 = dill.load(file)

###

with open(f'DDM/synthetic_data_test.pkl', 'rb') as file:
    synthetic_data = dill.load(file)

test_data = np.array([synth_data['choices'] for synth_data in synthetic_data])

###

with open(f'DDM/best_model_aic_2-10.pkl', 'rb') as file:
    model_aic_2to10 = dill.load(file)

with open(f'DDM/best_model_score_2-10.pkl', 'rb') as file:
    model_score_2to10 = dill.load(file)

# with open(f'DDM/best_model_score_2-20.pkl', 'rb') as file:
#     model_score_2to20 = dill.load(file)



###################
### Using model ###
###################

states_sequences = np.int16(test_data.reshape(-1,1))

states_sequences_lengths = [len(x) for x in test_data]

model_aic_2to10_score = model_aic_2to10.score(states_sequences, states_sequences_lengths)
model_score_2to10_score = model_score_2to10.score(states_sequences, states_sequences_lengths)

print(f'### Score ###\n AIC (2-10 states) : {model_aic_2to10_score}\n Score (2-10 states) {model_score_2to10_score}: ')

# model_score_2to20_score = model_score_2to20.score(states_sequences, states_sequences_lengths)

# states_sequences = []
# sequences_number = len(test_data)

# for i in range(sequences_number):
    
#     choices_sequence = test_data[i]
#     choices_number = len(test_data[i])
    
#     states_sequence = model.predict(np.int16(choices_sequence.reshape(-1,1)))
#     states_sequences.append(states_sequence)


def find_final_state_start(states_sequence,success_state):

    success_step = 0

    for i in range(len(states_sequence)):

        if states_sequence[i]==success_state:

            break

        success_step += 1

    return success_step

def find_threshold_cross(p_a_sequence,threshold):

    success_step = 0

    for i in range(len(p_a_sequence)):

        if p_a_sequence[i]>=threshold:

            break

        success_step += 1

    return success_step


final_state_start_list = []
threshold_cross_list = []

for i,s_seq in enumerate(states_sequences):

    final_state_start_list.append(find_final_state_start(s_seq,6))
    threshold_cross_list.append(find_threshold_cross(synthetic_data[i]['p_a'], 1.))

#############
### Plots ###
#############

### Residuals ###

fig=plt.figure(figsize=(4, 4), dpi=300, constrained_layout=False, facecolor='w')
gs = fig.add_gridspec(1, 1, hspace=0.5,)
row = gs[0,0].subgridspec(1, 1)

ax = plt.subplot(row[0,0])

bins = 40

data = [residuals_aic_2to10,residuals_score_2to10]

ax.hist(data, color=['blue', 'orange'], label=['AIC (2-10 states)', 'Score (2-10 states)'], bins=bins, alpha=0.5)

# ax.set_xlabel('Residuals of true and infered P(1) sequences')
ax.set_xlabel('sum((P_n-P_n_estimated)**2)/N')
ax.set_ylabel('Number of simulations')
ax.legend()

### Score ###
"""
fig=plt.figure(figsize=(4, 4), dpi=300, constrained_layout=False, facecolor='w')
gs = fig.add_gridspec(1, 1, hspace=0.5,)
row = gs[0,0].subgridspec(1, 1)

ax = plt.subplot(row[0,0])

bins = 40

data = [residuals_aic_2to10,residuals_score_2to10]

ax.hist(data, color=['blue', 'orange'], label=['AIC (2-10 states)', 'Score (2-10 states)'], bins=bins, alpha=0.5)

# ax.set_xlabel('Residuals of true and infered P(1) sequences')
ax.set_xlabel('sum((P_n-P_n_estimated)**2)/N')
ax.set_ylabel('Number of simulations')
ax.legend()
"""
plt.show()