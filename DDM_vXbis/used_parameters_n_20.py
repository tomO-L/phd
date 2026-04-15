import numpy as np


# Simulation generation parameters

P_CW_REWARD = 0.8
P_CCW_REWARD = 0
P_CW_INIT_RANGE = np.linspace(0.01,0.99,100)
STEPS_NUMBER = 40
NOISE_AMPLITUDE = 0.1
DRIFT_INIT = 0
DRIFT_VALUES_ARR = np.linspace(0.005,0.1,5)

DEFAULT_ARGS_DICT = {'p_cw_reward': P_CW_REWARD, 
             'p_ccw_reward': P_CCW_REWARD, 
             'p_cw_init': 0.5, 
             'steps_number': STEPS_NUMBER, 
             'noise_amplitude': NOISE_AMPLITUDE, 
             'drift_matrix': np.array([[DRIFT_VALUES_ARR[0], -DRIFT_VALUES_ARR[0]],
                                       [0                  , 0                   ]]), 
             'drift_init': DRIFT_INIT}


SIMULATIONS_SET_SIZE = 20
N_SETS = 100
SIMULATIONS_SET_SIZE_LIST = [SIMULATIONS_SET_SIZE]*N_SETS

# Fit parameters
STATES_NUMBER_RANGE = np.arange(2,12)
N_FITS = 50
N_JOBS = 4

# Other
MAIN_FOLDER_PATH = f'/home/david/Documents/code/loop_drift_estimation_output/n_{SIMULATIONS_SET_SIZE}'