#######################
### Import packages ###
#######################

from general_functions import *
from rbts_functions import *
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
# path_to_data_folder is the path of the folder where you store the folders of your different mice.
# path_to_data_folder='/LocalData/ForagingMice/4TowersTaskMethodPaper_Data/AurelienData/'
path_to_data_folder='/LocalData/ForagingMice/4TowersTaskMethodPaper_Data/MaudData/'

# Analysing the entire group of mice
# mice_to_analyse = [
#     "MOUEml1_5", "MOUEml1_8", "MOUEml1_11", "MOUEml1_12", "MOUEml1_13", "MOUEml1_15", "MOUEml1_18", "MOUEml1_20",
#     "MOURhoA_2", "MOURhoA_5", "MOURhoA_6", "MOURhoA_8", "MOURhoA_9", "MOURhoA_12", "MOURhoA_14",
#     "MOUB6NN_4", "MOUB6NN_6", "MOUB6NN_13", "MOUB6NN_15"
# ]

mice_to_analyse = ['MOU3974','MOU3975', 'MOU3987', 'MOU3988', 'MOU3991', 'MOU3992', 'MOU4551', 'MOU4552', 'MOU4560', 'MOU4561', 'MOU4562',
                   'MOU4563', 'MOU4623', 'MOU4964', 'MOU4965', 'MOU4986', 'MOU4987', 'MOU4988', 'MOU4993', 'MOU5007', 'MOU5008']

##################
### Parameters ###
##################

first_and_last_session_indexes = [0,20]


####################
### Computations ###
####################

# Load data


# print(f"{n_int_start_trap/n_rbts*100} % start in interior trapezes\n")
# print(f"{n_ext_start_trap/n_rbts*100} % start in exterior trapezes\n")
# print(f"{n_int_end_trap/n_rbts*100} % end in interior trapezes\n")
# print(f"{n_ext_end_trap/n_rbts*100} % end in exterior trapezes\n")


int_int_trap_ratio_persession = {mouse: [] for mouse in mice_to_analyse}
ext_ext_trap_ratio_persession = {mouse: [] for mouse in mice_to_analyse}
int_ext_trap_ratio_persession = {mouse: [] for mouse in mice_to_analyse}
ext_int_trap_ratio_persession = {mouse: [] for mouse in mice_to_analyse}

for mouse in mice_to_analyse:

    folder_path_mouse_to_process = os.path.join(path_to_data_folder, mouse)

    # Get the list of sessions for the current mouse
    sessions_to_process = sorted([name for name in os.listdir(folder_path_mouse_to_process)
                                  if os.path.isdir(os.path.join(folder_path_mouse_to_process, name))
                                  and name.startswith('MOU')])

    # limit the analysis to the subseet of session we want to analyse
    sessions_to_process = sessions_to_process[first_and_last_session_indexes[0]:first_and_last_session_indexes[1]]

    nb_sessions = len(sessions_to_process)
    print(f'Mouse {mouse}. There is/are {nb_sessions} sessions:')
    print(sessions_to_process, '\n')


    # Process each session for the current mouse
    for session_index, session_to_process in enumerate(sessions_to_process):

        
        # Define the pickle file path for the session
        output_pickle_filename = f"{session_to_process}_basic_processing_output.pickle"
        output_pickle_filepath = os.path.join(folder_path_mouse_to_process, session_to_process, output_pickle_filename)

        # Check if the pickle file exists
        if not os.path.exists(output_pickle_filepath):
            print(f'Pickle file does not exist for session {session_to_process}, skipping .....')
            # Append session Nan data to the respective dictionaries
            int_int_trap_ratio_persession[mouse].append([session_index + 1, np.nan])
            ext_ext_trap_ratio_persession[mouse].append([session_index + 1, np.nan])
            int_ext_trap_ratio_persession[mouse].append([session_index + 1, np.nan])
            ext_int_trap_ratio_persession[mouse].append([session_index + 1, np.nan])
            continue  # Skip to the next session if the pickle file does not exist
        
        data = load_pickle_data(folder_path_mouse_to_process,session_index)

        rbts = data['all_epochs']['run_between_towers']

        n_rbts = len(rbts)

        extremities_list = compute_rbt_extremities(rbts)

        int_int_trap_ratio = extremities_list.count('intint')/n_rbts
        int_ext_trap_ratio = extremities_list.count('intext')/n_rbts
        ext_int_trap_ratio = extremities_list.count('extint')/n_rbts
        ext_ext_trap_ratio = extremities_list.count('extext')/n_rbts

        int_int_trap_ratio_persession[mouse].append([session_index + 1, int_int_trap_ratio])
        int_ext_trap_ratio_persession[mouse].append([session_index + 1, int_ext_trap_ratio])
        ext_int_trap_ratio_persession[mouse].append([session_index + 1, ext_int_trap_ratio])
        ext_ext_trap_ratio_persession[mouse].append([session_index + 1, ext_ext_trap_ratio])
            



#############
### Plots ###
#############

fig=plt.figure(figsize=(4, 7), dpi=300, constrained_layout=False, facecolor='w')
gs = fig.add_gridspec(1, 1)
row1 = gs[0].subgridspec(2, 2)
ax1 = plt.subplot(row1[0,0])
ax2 = plt.subplot(row1[0,1])
ax3 = plt.subplot(row1[1,0])
ax4 = plt.subplot(row1[1,1])

plot_learning_curve(int_int_trap_ratio_persession, mice_to_analyse, ax1)
ax1.set_ylabel('int to int rbt')
ax1.set_ylim([0,1])
ax1.axhline(0.5,0,20, color='grey', linestyle='--')

plot_learning_curve(int_ext_trap_ratio_persession, mice_to_analyse, ax2)
ax2.set_ylabel('int to ext rbt')
ax2.set_ylim([0,1])
ax2.axhline(0.5,0,20, color='grey', linestyle='--')

plot_learning_curve(ext_int_trap_ratio_persession, mice_to_analyse, ax3)
ax3.set_ylabel('ext to int rbt')
ax3.set_ylim([0,1])
ax3.axhline(0.5,0,20, color='grey', linestyle='--')

plot_learning_curve(ext_ext_trap_ratio_persession, mice_to_analyse, ax4)
ax4.set_ylabel('ext to ext rbt')
ax4.set_ylim([0,1])
ax4.axhline(0.5,0,20, color='grey', linestyle='--')



plt.show()