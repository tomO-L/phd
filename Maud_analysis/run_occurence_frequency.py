#######################
### Import packages ###
#######################

from general_functions import *
import matplotlib.pyplot as plt
from matplotlib import colormaps
from hmmlearn import hmm, vhmm
import sys
import time
from matplotlib import cm
from matplotlib.colors import Normalize
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(call_pdb=False)

plt.style.use('paper.mplstyle')

start_time = time.time()

######################
### Mice selection ###
######################

# defining data folder path and mice list

path_to_data_folder='/LocalData/ForagingMice/4TowersTaskMethodPaper_Data/Group2Data/'

mice_to_analyse = ['MOU3974','MOU3975', 'MOU3987', 'MOU3988', 'MOU3991', 'MOU3992', 'MOU4551', 'MOU4552', 'MOU4560', 'MOU4561', 'MOU4562',
                   'MOU4563', 'MOU4623', 'MOU4964', 'MOU4965', 'MOU4986', 'MOU4987', 'MOU4988', 'MOU4993', 'MOU5007', 'MOU5008']

mouse = mice_to_analyse[10]

folder_path_mouse_to_analyse = os.path.join(path_to_data_folder, mouse)

##################
### Parameters ###
##################

session_index = 14

####################
### Computations ###
####################

# Load data
data = load_pickle_data(folder_path_mouse_to_analyse,session_index)
# Order epochs

ordered_runs, ordered_runs_types_number, ordered_runs_frames = extract_runs_sequence(path_to_data_folder, mouse, session_index)

run_sequence = []

for run in ordered_runs:
    
    run_type = identify_action(run)[0]

    run_sequence.append(run_type)

#############
### Plots ###
#############

fig=plt.figure(figsize=(5, 5), dpi=300, constrained_layout=False, facecolor='w')
gs = fig.add_gridspec(1, 1)
row1 = gs[0].subgridspec(1, 1)
ax = plt.subplot(row1[0])

plot_states_occurence_frequency(run_sequence, ax, window_size=50)
plot_reward_rate(ordered_runs,ax, window_size=50)

ax.set_title(f'{mouse}, session {session_index}')
ax.legend()

fig.tight_layout()

plt.show()

