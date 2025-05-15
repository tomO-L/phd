#######################
### Import packages ###
#######################

from functions import *
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
#path_to_data_folder='/LocalData/ForagingMice/4TowersTaskMethodPaper_Data/AurelienData/'
path_to_data_folder='/LocalData/ForagingMice/4TowersTaskMethodPaper_Data/MaudData/'

# Analysing the entire group of mice
# mice_to_analyse = [
#     "MOUEml1_5", "MOUEml1_8", "MOUEml1_11", "MOUEml1_12", "MOUEml1_13", "MOUEml1_15", "MOUEml1_18", "MOUEml1_20",
#     "MOURhoA_2", "MOURhoA_5", "MOURhoA_6", "MOURhoA_8", "MOURhoA_9", "MOURhoA_12", "MOURhoA_14",
#     "MOUB6NN_4", "MOUB6NN_6", "MOUB6NN_13", "MOUB6NN_15"
# ]

mice_to_analyse = ['MOU3974','MOU3975', 'MOU3987', 'MOU3988', 'MOU3991', 'MOU3992', 'MOU4551', 'MOU4552', 'MOU4560', 'MOU4561', 'MOU4562',
                   'MOU4563', 'MOU4623', 'MOU4964', 'MOU4965', 'MOU4986', 'MOU4987', 'MOU4988', 'MOU4993', 'MOU5007', 'MOU5008']

mouse = mice_to_analyse[0]

##################
### Parameters ###
##################

folder_path_mouse_to_analyse = os.path.join(path_to_data_folder, mouse)

session_index = 19

####################
### Computations ###
####################

# Load data
data = load_pickle_data(folder_path_mouse_to_analyse,session_index)
# Order epochs
# ordered_runs, ordered_runs_frames = order_runs(data['all_epochs'])
ordered_runs, ordered_runs_types_number, ordered_runs_frames = extract_runs_sequence(path_to_data_folder, mouse, session_index)

# Define color mapping for TATs
mapping = []

for run in ordered_runs:

    if run[0]=='run_around_tower':

        direction = run[4]['direction']

        if direction=='CW':

            mapping.append(1)

        elif direction=='CCW':

            mapping.append(-1)

        else:

            print('ERROR: Invalid direction string')
    else:

        mapping.append(0)

cmap = plt.cm.berlin
norm = Normalize(vmin=-1, vmax=1)

# turns_per_rewarded_visit = np.where(np.logical_not(np.equal(rewarded_turns_per_visit,0)),turns_per_visit,np.nan)
# turns_per_rewarded_visit = np.where(np.logical_not(np.isnan(rewarded_turns_per_visit)),turns_per_rewarded_visit,np.nan)
# turns_vs_maxreward_per_rewarded_visit = np.where(np.logical_not(np.equal(rewarded_turns_per_visit,0)),turns_per_visit-max_rewards,np.nan)

#############
### Plots ###
#############

fig=plt.figure(figsize=(4, 7), dpi=300, constrained_layout=False, facecolor='w')
gs = fig.add_gridspec(1, 1)
row1 = gs[0].subgridspec(1, 1)
ax1 = plt.subplot(row1[0])

plot_runs_sequence(ax1,ordered_runs_types_number, cmaps=cmap(norm(mapping)))

# plot = ax1.scatter(visits_time, numcoded_first_direction_per_visit, linewidth=1, marker='|', c=cmap(norm(turns_per_visit)))
# ax1.plot(visits_time_rewarded_visits,numcoded_first_direction_per_rewarded_visit, linewidth=0.2, marker='|')

# ax1.set_yticks([0,1],['CW','CCW'])
# ax1.set_ylim([-0.5,1.5])
cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax1, extend='both')

# ax1.plot(visits_time, rewarded_turns_per_visit, marker='o', markersize=1)
# ax1.plot(visits_time, turns_per_visit-rewarded_turns_per_visit, marker='o', markersize=1)


plt.show()

