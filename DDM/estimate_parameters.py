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

#################
### Functions ###
#################

def run_sequence(p_a, p_a_reward, steps_number, noise_amplitude, delta, drift):

    p_b = 1 - p_a
    p_b_reward = 1 - p_a_reward

    reward_sequence = []
    choice_sequence = []
    p_a_sequence = []
    drift_sequence = []

    for _ in range(steps_number):


        ### Choice
        choice = np.random.choice([1,0], p=[p_a, p_b])

        ### Reward
        if choice == 1:

            reward = np.random.choice([1,0], p=[p_a_reward, 1-p_a_reward])

        else:

            reward = np.random.choice([1,0], p=[p_b_reward, 1-p_b_reward])

        
        ### Storage
        reward_sequence.append(reward)
        choice_sequence.append(choice)
        p_a_sequence.append(p_a)
        drift_sequence.append(drift)

        ### Probability update
        noise = np.random.randn() * noise_amplitude

        drift = reward*delta

        p_a = p_a + drift + noise
        
        if p_a<0:
            p_a = 0
        elif p_a>1:
            p_a = 1

        p_b = 1 - p_a

    ddm_result = {'rewards': np.array(reward_sequence), 'choices': np.array(choice_sequence), 'p_a': np.array(p_a_sequence), 'drift': np.array(drift_sequence)}

    return ddm_result

def compute_simulations_average(p_a, p_a_reward, steps_number, noise_amplitude, delta, drift, n_simulations=300):

    synthetic_proba_list = []

    for _ in tqdm(range(n_simulations), leave=False):
    
        ddm_result = run_sequence(p_a, p_a_reward, steps_number, noise_amplitude, delta, drift)

        synthetic_proba_list.append(ddm_result['p_a'])

    average_trajectory = np.mean(synthetic_proba_list,axis=0)

    return average_trajectory

def compute_mean_square_error(delta, args):
    
    p_a = args[0]
    p_a_reward = args[1]
    steps_number = args[2]
    noise_amplitude = args[3]
    drift = args[4]
    n_simulations = args[5]
    reconstructed_average_trajectory = args[6]

    average_trajectory = compute_simulations_average(p_a, p_a_reward, steps_number, noise_amplitude, delta, drift, n_simulations=n_simulations)

    mse = (np.sum((average_trajectory - reconstructed_average_trajectory))**2)/steps_number

    return mse

###################
### Import Data ###
###################

#with open(f'DDM/reconstructed_average_p_a.pkl', 'rb') as file:
#    reconstructed_average_p_a = dill.load(file)

with open(f'DDM/simple_synthetic_data_test2.pkl', 'rb') as file:
    synthetic_data = dill.load(file)

test_data = [synth_data['choices'] for synth_data in synthetic_data][20]

with open(f'DDM/simple_best_model_score_2-40.pkl', 'rb') as file:
    model = dill.load(file)


###############
### Use HMM ###
###############

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


###########
### Fit ###
###########

args = [p_a, p_a_reward, steps_number, noise_amplitude, drift, 5000, reconstructed_average_p_a]

# opt_res_list = []

# for _ in range(20):
    
#     opt_res = opt.minimize(compute_mean_square_error, 0.01, args=args, method= 'Powell')
#     opt_res_list.append(opt_res)
#     print(opt_res.x)

# print(opt_res_list)

delta_range = np.linspace(0.01,0.1,250)

# mse_list = []

# for delta in tqdm(delta_range):

#     mse_list.append(compute_mean_square_error(delta, args))

# min_mse = np.min(mse_list)
# recovered_delta = delta_range[np.where(mse_list==min_mse)[0]]

end_time = time.time()
print(f"Ca a pris {(end_time-start_time)//60} min {(end_time-start_time)%60} s")

####################
### Run and Plot ###
####################

# fig=plt.figure(figsize=(4, 7), dpi=300, constrained_layout=False, facecolor='w')
# gs = fig.add_gridspec(1, 1, hspace=0.5,)
# row = gs[0,0].subgridspec(3, 1)

# ax = plt.subplot(row[:])

# ax.plot(delta_range,mse_list, label="Mean Square Error for different Drift values")
# ax.axvline(0.05, linewidth=0.7, color='k', linestyle='--', label='Drift used to generate simulations')
# ax.axvline(recovered_delta, linewidth=0.7,color='grey', linestyle='--', label='Recovered Drift (i.e with minimum MSE)')

# ax.set_xlabel('Delta')
# ax.set_ylabel('Mean Square Error')

# ax.legend()

# plt.show()



fig=plt.figure(figsize=(4, 7), dpi=300, constrained_layout=False, facecolor='w')
gs = fig.add_gridspec(1, 1, hspace=0.5,)
row = gs[0,0].subgridspec(3, 1)

ax = plt.subplot(row[:])

steps = np.arange(steps_number)

delta_range = np.linspace(0.01,0.1,10)


for delta in tqdm(delta_range):

   mean_trajectory = compute_simulations_average(p_a, p_a_reward, steps_number, noise_amplitude, delta, drift, n_simulations=1000)

   ax.plot(steps, mean_trajectory, alpha=0.7)
   ax.text(steps[-1],mean_trajectory[-1], f'drift = {np.round(delta,3)}', fontsize=5)


ax.set_xlabel('Steps')
ax.set_ylabel('Average probability to chose 1')

ax.set_xticks(steps)

ax.set_ylim([0,1])

plt.show()






