#######################
### Import packages ###
#######################

import matplotlib.pyplot as plt
import sys
import time
from IPython.core import ultratb
import dill
import numpy as np
sys.excepthook = ultratb.FormattedTB(call_pdb=False)

plt.style.use('paper.mplstyle')

# Time counter
start_time = time.time()

##################
### Parameters ###
##################

steps_number = 40
noise_amplitude = 0.1
delta = 0.1
drift = 0.01
p_a = 0.5
p_a_reward = 1

# np.random.seed(587) # initial seed
np.random.seed(58777) # test seed

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

    for i in range(steps_number):

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

        drift = drift*(1 + reward_sequence[i-1]*delta)
        # print(drift)
        p_a = p_a + drift + noise
        
        if p_a<0:
            p_a = 0
        elif p_a>1:
            p_a = 1

        p_b = 1 - p_a

    ddm_result = {'rewards': np.array(reward_sequence), 'choices': np.array(choice_sequence), 'p_a': np.array(p_a_sequence), 'drift': np.array(drift_sequence)}

    return ddm_result


####################
### Run and Plot ###
####################


fig=plt.figure(figsize=(4, 7), dpi=300, constrained_layout=False, facecolor='w')
gs = fig.add_gridspec(1, 1, hspace=0.5,)
row = gs[0,0].subgridspec(3, 1)

steps = np.arange(steps_number)
synthetic_data = []

show_plot = False

for _ in range(5000):
    

    # p_a = np.random.rand()

    ddm_result = run_sequence(p_a, p_a_reward, steps_number, noise_amplitude, delta, drift)

    synthetic_data.append(ddm_result)

    if not show_plot:

        continue

    reward_sequence = ddm_result['rewards']
    choice_sequence = ddm_result['choices']
    p_a_sequence = ddm_result['p_a']

    ax1 = plt.subplot(row[0,0])
    ax1.step(steps, reward_sequence, label='Reward Sequence', alpha=0.3)
    ax1.set_ylabel('Reward')
    ax1.set_xticks(steps)
    ax1.set_yticks([0,1])

    ax2 = plt.subplot(row[1,0])
    ax2.step(steps, choice_sequence, label='Choice Sequence', alpha=0.3)
    ax2.set_ylabel('Choice')
    ax2.set_xticks(steps)
    ax1.set_yticks([0,1])

    ax3 = plt.subplot(row[2,0])
    ax3.plot(steps, p_a_sequence, label='Probability Sequence of A', alpha=0.3)
    ax3.set_ylabel('Probability\nto chose A')
    ax3.set_xticks(steps)
    ax3.set_ylim([-0.05,1.05])

plt.show()

"""
for i in range(steps_number):

    ### Choice
    choice = np.random.choice(['A','B'], p=[p_a, p_b])

    ### Reward
    if choice == 'A':

        reward = np.random.choice([1,0], p=[p_a_reward, 1-p_a_reward])

    else:

        reward = np.random.choice([1,0], p=[p_b_reward, 1-p_b_reward])

    ### Storage
    reward_sequence.append(reward)
    choice_sequence.append(choice)
    p_a_sequence.append(p_a)

    ### Probability update
    noise = np.random.randn() * noise_amplitude

    drift = drift*(1 + reward_sequence[i-1]*delta)
    print(drift)
    p_a = p_a + drift + noise
    
    if p_a<0:
        p_a = 0
    elif p_a>1:
        p_a = 1

    p_b = 1 - p_a
"""

save = True
if save:
    with open(f'DDM/synthetic_data_test.pkl', 'wb') as file:
        dill.dump(synthetic_data, file)

# Time counter
end_time = time.time()
print(f"Ca a pris {(end_time-start_time)//60} min {(end_time-start_time)%60} s")


