import numpy as np
from tqdm import tqdm


def run_simulation(p_a, p_a_reward, p_b_reward, steps_number, noise_amplitude, delta, drift):

    p_b = 1 - p_a

    p_a_0 = p_a

    reward_sequence = []
    choice_sequence = []
    p_a_sequence = []
    drift_sequence = []

    for _ in range(steps_number):

        ### Action
        action = np.random.choice([1,0], p=[p_a, p_b])

        ### Reward
        if action == 1:

            reward = np.random.choice([1,0], p=[p_a_reward, 1-p_a_reward])

        else:

            reward = np.random.choice([1,0], p=[p_b_reward, 1-p_b_reward])

        ### Storage
        reward_sequence.append(reward)
        choice_sequence.append(action)
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

    ddm_result = {'parameters': [p_a_0, p_a_reward, p_b_reward, steps_number, noise_amplitude, delta, drift],'rewards': np.array(reward_sequence), 'choices': np.array(choice_sequence), 'p_a': np.array(p_a_sequence), 'drift': np.array(drift_sequence)}

    return ddm_result

def run_simulations_batch(p_a, p_a_reward, p_b_reward, steps_number, noise_amplitude, delta, drift, n_simulations):

    simulations_batch = []

    for _ in tqdm(range(n_simulations)):
    
        ddm_result = run_simulation(p_a, p_a_reward, p_b_reward, steps_number, noise_amplitude, delta, drift)

        simulations_batch.append(ddm_result)

    return simulations_batch

