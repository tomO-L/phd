### Import packages ###

from processing_TowerCoordinates import *
from processing_session_trajectory import *
from functions import *
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.path as mpath
import matplotlib.cm as cm
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import hmmlearn

import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(call_pdb=False)


### Mice selection ###

# defining data folder path and mice list
# path_to_data_folder is the path of the folder where you store the folders of your different mice.
path_to_data_folder='/LocalData/ForagingMice/4TowersTaskMethodPaper_Data/AurelienData/'

# Analysing the entire group of mice
mice_to_analyse = [
    "MOUEml1_5", "MOUEml1_8", "MOUEml1_11", "MOUEml1_12", "MOUEml1_13", "MOUEml1_15", "MOUEml1_18", "MOUEml1_20",
    "MOURhoA_2", "MOURhoA_5", "MOURhoA_6", "MOURhoA_8", "MOURhoA_9", "MOURhoA_12", "MOURhoA_14",
    "MOUB6NN_4", "MOUB6NN_6", "MOUB6NN_13", "MOUB6NN_15"
]

# Verify that all folders in mice_to_analyse are present in path_to_data_folder
missing_folders = [mouse for mouse in mice_to_analyse if not os.path.isdir(os.path.join(path_to_data_folder, mouse))]

if missing_folders:
    print("Missing mice folders:", missing_folders)
else:
    print("All mice folders are present in data folder.")

# Print the number of mice, the list of mice
#print(f' {len(mice_to_analyse)} {"mice" if len(mice_to_analyse) > 1 else "mouse"} will be analysed\n')

mouse = mice_to_analyse[0]

### Parameters ###

arena_coordinates_cm=[[2.5, 91.7], [90.3, 91.7], [90.3, 2.7], [2.5, 2.7]]
mouse_folder_path = os.path.join(path_to_data_folder, mouse)



### Extract epochs ###

data = load_data(mouse_folder_path,6)

ordered_epochs, ordered_epochs_frames = order_epochs(data['all_epochs'])

print(ordered_epochs[38])




# Get the positions
# positions = np.array(session_data['positions'])
# times_videoFrames = np.array(session_data['timeofframes'])


# plot_trajectory_on_maze(arena_coordinates, tower_coordinates, all_trapezes_coordinates, reward_spouts_coordinates,
#                             xpositions, ypositions, time_video_frames, chunk_start=None, chunk_end=None,
#                             towerscolor=['k', 'k', 'k', 'k'], showtowerID=True, showspoutID=False, showdrops=True, show_arena_size=False, ax=None)


