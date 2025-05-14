#######################
### Import packages ###
#######################

from functions import *
import matplotlib.pyplot as plt
from matplotlib import colormaps
from hmmlearn import hmm, vhmm
import sys
import time
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

#################
### Functions ###
#################

def find_tat_by_time(tats,time):

    """
    Find a TAT by its occurence time
    """

    for tat in tats:

        if tat[4]['epoch_time']==time:
            
            return tat
        
    print("WARNING: TAT not found")

def compute_first_guess(folder_path_mouse_to_analyse,session_index):
    
    _, rewarded_turns_per_visit, _, _ = compute_turns_per_rewarded_visit(folder_path_mouse_to_analyse, session_index)

    # Load data
    data = load_pickle_data(folder_path_mouse_to_analyse,session_index)
    # Find visits
    visits = find_visits(data['all_epochs'])


    first_direction_per_visit = []
    first_result_per_visit = []
    visits_time = []

    for visit in visits:
        
        visit_time = visit['visit_time']
        visits_time.append(visit_time)

        first_turn = find_tat_by_time(data['all_epochs']['run_around_tower'],visit_time)
        first_direction_per_visit.append(first_turn[3]['direction'])
        first_result_per_visit.append(first_turn[3]['Rewarded'])

    numcoded_first_direction_per_visit = np.where(np.array(first_direction_per_visit)=='CW',1,0)
    numcoded_first_result_per_visit = np.int8(np.array(first_result_per_visit))
    numcoded_rewarded_visit = np.where(rewarded_turns_per_visit!=0,1,0)

    return numcoded_first_direction_per_visit, numcoded_first_result_per_visit, numcoded_rewarded_visit, visits_time

####################
### Computations ###
####################

ratio_rewarded_first_turn_pervisit_persessions = {mouse: [] for mouse in mice_to_analyse}
rewarded_visits_persessions = {mouse: [] for mouse in mice_to_analyse}

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
            ratio_rewarded_first_turn_pervisit_persessions[mouse].append([session_index + 1, np.nan])
            rewarded_visits_persessions[mouse].append([session_index + 1, np.nan])
            continue  # Skip to the next session if the pickle file does not exist

        _, numcoded_first_result_per_visit, numcoded_rewarded_visit, _ = compute_first_guess(folder_path_mouse_to_process, session_index)

        ratio_rewarded_first_turn_pervisit_persessions[mouse].append([session_index + 1, np.sum(numcoded_first_result_per_visit)/len(numcoded_first_result_per_visit)])
        rewarded_visits_persessions[mouse].append([session_index + 1, np.sum(numcoded_rewarded_visit)])


#############
### Plots ###
#############

fig=plt.figure(figsize=(4, 7), dpi=300, constrained_layout=False, facecolor='w')
gs = fig.add_gridspec(1, 1)
row1 = gs[0].subgridspec(1, 1)
ax1 = plt.subplot(row1[0])

all_mice_values_persessions = []

for mouse in mice_to_analyse:

    values_persessions = np.transpose(ratio_rewarded_first_turn_pervisit_persessions[mouse])
    all_mice_values_persessions.append(values_persessions[1])

    ax1.plot(values_persessions[0],values_persessions[1], alpha=0.3)

median_values = np.nanmedian(all_mice_values_persessions, axis=0)
upper_quartile_values = np.nanpercentile(all_mice_values_persessions, 75, axis=0)
lower_quartile_values = np.nanpercentile(all_mice_values_persessions, 25, axis=0)

# day_diff = []

# for i in np.arange(1,len(median_values), step=2):

#     res = median_values[i]/median_values[i-1]

#     day_diff.append(res)

print([median_values-lower_quartile_values, upper_quartile_values-median_values])

ax1.errorbar(values_persessions[0],median_values, yerr=[median_values-lower_quartile_values, upper_quartile_values-median_values], color='black')
ax1.axhline(0.5,0,20, color='grey', linestyle='--')

# ax1.plot(np.arange(1,len(median_values),step=2)+1, day_diff, color='grey')

ax1.set_xticks(values_persessions[0])


#ax1.scatter(visits_time,numcoded_first_direction_per_visit, linewidth=1, marker='|', c=cmap(norm(turns_per_visit-rewarded_turns_per_visit)))
# ax1.plot(visits_time_rewarded_visits,numcoded_first_direction_per_rewarded_visit, linewidth=0.2, marker='|')

# ax1.set_yticks([0,1],['CW','CCW'])
# ax1.set_ylim([-0.5,1.5])
# ax1.plot(visits_time, rewarded_turns_per_visit, marker='o', markersize=1)
# ax1.plot(visits_time, turns_per_visit-rewarded_turns_per_visit, marker='o', markersize=1)


plt.show()

