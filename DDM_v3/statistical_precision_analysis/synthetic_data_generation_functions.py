import numpy as np
import copy
from tqdm.notebook import tqdm

# def run_simulation(p_a, p_a_reward, p_b_reward, steps_number, noise_amplitude, reward_drift, failure_drift, drift_init):

#     p_b = 1 - p_a

#     p_a_init = p_a

#     drift = drift_init

#     reward_sequence = []
#     choice_sequence = []
#     p_a_sequence = []
#     drift_sequence = []

#     for _ in range(steps_number):

#         ### Action
#         action = np.random.choice([1,0], p=[p_a, p_b])

#         ### Reward
#         if action == 1:

#             reward = np.random.choice([1,0], p=[p_a_reward, 1-p_a_reward])

#         else:

#             reward = np.random.choice([1,0], p=[p_b_reward, 1-p_b_reward])

#         ### Storage
#         reward_sequence.append(reward)
#         choice_sequence.append(action)
#         p_a_sequence.append(p_a)
#         drift_sequence.append(drift)

#         ### Probability update
#         noise = np.random.randn() * noise_amplitude

#         # drift_value = a if reward == 1 else b
#         # drift_sign = 1 if action == 1 else -1
        
#         drift_matrix = np.array([[+ failure_drift , -reward_drift],
#                                  [- failure_drift , reward_drift]])
        
#         drift = drift_matrix[action,reward]
        
#         """
#                                                                  Rewarded or not
#                     |-----------------------------------------------------|----------------------------------------------------|
#                     | action = 0 (CCW) and reward = 1 --> + failure_drift | action = 0 (CCW) and reward = 0 --> - reward_drift |
#         Action type |-----------------------------------------------------|----------------------------------------------------|
#                     | action = 1 (CW)  and reward = 0 --> - failure_drift | action = 1 (CW)  and reward = 1 --> + reward_drift |
#                     |-----------------------------------------------------|----------------------------------------------------|
#         """

#         p_a = p_a + drift + noise
        
#         if p_a<0:
#             p_a = 0
#         elif p_a>1:
#             p_a = 1

#         p_b = 1 - p_a

#     ddm_result = {'parameters': [p_a_init, p_a_reward, p_b_reward, steps_number, noise_amplitude, reward_drift, failure_drift, drift_init],'rewards': np.array(reward_sequence), 'choices': np.array(choice_sequence), 'p_a': np.array(p_a_sequence), 'drift': np.array(drift_sequence)}

#     return ddm_result

# def run_simulations_batch(p_a, p_a_reward, p_b_reward, steps_number, noise_amplitude, reward_drift, failure_drift, drift_init, n_simulations):

#     simulations_batch = []

#     for _ in tqdm(range(n_simulations)):
    
#         ddm_result = run_simulation(p_a, p_a_reward, p_b_reward, steps_number, noise_amplitude,  reward_drift, failure_drift, drift_init)

#         simulations_batch.append(ddm_result)

#     return simulations_batch







def run_simulation(args_dict):

    p_cw_init = args_dict['p_cw_init']
    p_cw_reward = args_dict['p_cw_reward']
    p_ccw_reward = args_dict['p_ccw_reward']
    steps_number = args_dict['steps_number']
    noise_amplitude = args_dict['noise_amplitude']
    drift_matrix = args_dict['drift_matrix']
    # reward_drift = args_dict['reward_drift']
    # failure_drift = args_dict['failure_drift']    
    drift_init = args_dict['drift_init']

    p_cw = p_cw_init
    p_ccw = 1 - p_cw

    drift = drift_init

    reward_sequence = []
    choice_sequence = []
    p_cw_sequence = []
    drift_sequence = []

    for _ in range(steps_number):

        ### Action
        action = np.random.choice([1,0], p=[p_cw, p_ccw])

        ### Reward
        if action == 1:

            reward = np.random.choice([1,0], p=[p_cw_reward, 1-p_cw_reward])

        else:

            reward = np.random.choice([1,0], p=[p_ccw_reward, 1-p_ccw_reward])

        ### Storage
        reward_sequence.append(reward)
        choice_sequence.append(action)
        p_cw_sequence.append(p_cw)
        drift_sequence.append(drift)

        ### Probability update
        noise = np.random.randn() * noise_amplitude

        # drift_value = a if reward == 1 else b
        # drift_sign = 1 if action == 1 else -1
        
        # drift_matrix = np.array([[+ failure_drift , -reward_drift],
        #                          [- failure_drift , reward_drift]])
        
        # drift = drift_matrix[action,reward]
        drift = np.matmul(np.array([reward,1-reward]), drift_matrix)
        drift = np.matmul(np.array([action,1-action]), drift)

        """
        BON C'EST FAUX MAIS TQT
                                                                 Rewarded or not
                    |-----------------------------------------------------|----------------------------------------------------|
                    | action = 0 (CCW) and reward = 1 --> + failure_drift | action = 0 (CCW) and reward = 0 --> - reward_drift |
        Action type |-----------------------------------------------------|----------------------------------------------------|
                    | action = 1 (CW)  and reward = 0 --> - failure_drift | action = 1 (CW)  and reward = 1 --> + reward_drift |
                    |-----------------------------------------------------|----------------------------------------------------|
        """

        p_cw = p_cw + drift + noise
        
        if p_cw<0:
            p_cw = 0
        elif p_cw>1:
            p_cw = 1

        p_ccw = 1 - p_cw

    ddm_result = {'parameters': args_dict, 'rewards': np.array(reward_sequence), 'choices': np.array(choice_sequence), 'p_cw': np.array(p_cw_sequence), 'drift': np.array(drift_sequence)}

    return ddm_result

def run_simulations_batch(args_dict, n_simulations, p_cw_init_sequence=None):

    simulations_batch = []


    for i in tqdm(range(n_simulations)):
    
        if p_cw_init_sequence:

            args_dict['p_cw_init'] = p_cw_init_sequence[i]

    
        ddm_result = run_simulation(args_dict)

        simulations_batch.append(ddm_result)

    return simulations_batch

def run_simulations_batch_switch(args_dict, n_simulations, p_cw_init_sequence=None):

    simulations_batch = []

    p_cw_reward = args_dict['p_cw_reward']
    p_ccw_reward = args_dict['p_ccw_reward']

    for i in tqdm(range(n_simulations)):
    
        temp_args_dict = copy.deepcopy(args_dict)

        if p_cw_init_sequence:

            temp_args_dict['p_cw_init'] = p_cw_init_sequence[i]

        if i>=int(n_simulations/2):

            temp_args_dict['p_cw_reward'], temp_args_dict['p_ccw_reward'] = p_ccw_reward, p_cw_reward

        ddm_result = run_simulation(temp_args_dict)

        simulations_batch.append(ddm_result)

    return simulations_batch