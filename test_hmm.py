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
from hmmlearn import hmm

import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(call_pdb=False)

plt.style.use('paper.mplstyle')


### Mice selection ###

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


# Verify that all folders in mice_to_analyse are present in path_to_data_folder
missing_folders = [mouse for mouse in mice_to_analyse if not os.path.isdir(os.path.join(path_to_data_folder, mouse))]

if missing_folders:
    print("Missing mice folders:", missing_folders)
else:
    print("All mice folders are present in data folder.")

# Print the number of mice, the list of mice
#print(f' {len(mice_to_analyse)} {"mice" if len(mice_to_analyse) > 1 else "mouse"} will be analysed\n')

####################################
mouse = mice_to_analyse[0]
session_index = 19
####################################

### Parameters ###

mouse_folder_path = os.path.join(path_to_data_folder, mouse)
epoch_types = ['run_around_tower', 'run_between_towers', 'run_toward_tower', 'exploratory_run']

### Extract epochs ###

data = load_data(mouse_folder_path, session_index)

ordered_epochs, ordered_epochs_frames = order_epochs(data['all_epochs'])

ordered_epochs_types = []

for e in ordered_epochs:

    ordered_epochs_types.append(e[0])

while 'immobility' in ordered_epochs_types:

    ordered_epochs_types.remove('immobility')

num_epoch = len(ordered_epochs_types)

### Visualize data ###

plot = True

if plot:

    fig=plt.figure(figsize=(4, 4), dpi=300, constrained_layout=False, facecolor='w')
    gs = fig.add_gridspec(1, 1)
    row1 = gs[0].subgridspec(2, 1)

    # Epochs distributions
    ax1 = plt.subplot(row1[0])

    epoch_types_ditribution = [
                                ordered_epochs_types.count(epoch_types[0]),
                                ordered_epochs_types.count(epoch_types[1]),
                                ordered_epochs_types.count(epoch_types[2]),
                                ordered_epochs_types.count(epoch_types[3]),
    ]

    ax1.bar(epoch_types, epoch_types_ditribution)

    # Epoch sequence
    ax2 = plt.subplot(row1[1])

    for i, e_type in enumerate(epoch_types):

        ordered_epochs_type_number = np.where(np.array(ordered_epochs_types)==e_type, np.arange(num_epoch), np.nan)

        x_barh = np.transpose([ordered_epochs_type_number,np.ones(num_epoch)])
        y_barh = [i-0.25,0.5]

        ax2.broken_barh(x_barh, y_barh)

    ax2.set_yticks(np.arange(len(epoch_types)), epoch_types)

    # Display figure
    plt.show()




















### Infer model ###

ordered_epochs_types_number = np.nan * np.ones(num_epoch)

for i, e_type in enumerate(epoch_types):
    ordered_epochs_types_number = np.where(np.array(ordered_epochs_types)==e_type, i, ordered_epochs_types_number)


emissions = np.int8(ordered_epochs_types_number.reshape(-1,1))

# split our data into training and validation sets (50/50 split)
X_train = emissions[:emissions.shape[0] // 2]
X_validate = emissions[emissions.shape[0] // 2:]



# check optimal score

best_score = best_model = None
n_fits = 50
np.random.seed(13)

for n in (2,3,4):
    for idx in range(n_fits):
        model = hmm.CategoricalHMM(
            n_components=n, random_state=idx,
            init_params='se', n_features=4)  # don't init transition, set it below
        # we need to initialize with random transition matrix probabilities
        # because the default is an even likelihood transition
        # we know transitions are rare (otherwise the casino would get caught!)
        # so let's have an Dirichlet random prior with an alpha value of
        # (0.1, 0.9) to enforce our assumption transitions happen roughly 10%
        # of the time

        transmat = []
        for _ in range(n):

            row = np.random.uniform(size=n)
            row = row/np.sum(row)
            
            transmat.append(row)

        transmat = np.array(transmat)
        model.transmat_ = transmat
        
        model.fit(X_train)
        score = model.score(X_validate)
        print(f'Model {n} components #{idx}\tScore: {score}')
        if best_score is None or score > best_score:
            best_model = model
            best_score = score

# use the Viterbi algorithm to predict the most likely sequence of states
# given the model
states = best_model.predict(emissions)

print(f'Best score:      {best_score}')

# plot our recovered states compared to generated (aim 1)
fig, ax = plt.subplots()
#ax.plot(gen_states[:500], label='generated')
ax.plot(abs(states-1) + 1.5, label='recovered')
ax.set_yticks([])
ax.set_title('States compared to generated')
ax.set_xlabel('Time (# rolls)')
ax.legend()
plt.show()

print(f'Transmission Matrix Recovered:\n{best_model.transmat_.round(3)}\n\n')

print(f'Emission Matrix Recovered:\n{best_model.emissionprob_.round(3)}\n\n')

# print(data['all_epochs'].keys())


















gen_epoch_types, gen_states = best_model.sample(num_epoch)

if plot:

    fig=plt.figure(figsize=(4, 4), dpi=300, constrained_layout=False, facecolor='w')
    gs = fig.add_gridspec(1, 1)
    row1 = gs[0].subgridspec(2, 1)

    # Epochs distributions
    ax1 = plt.subplot(row1[0])

    gen_epoch_types_ditribution = [
                                np.count_nonzero(gen_epoch_types.reshape(1,-1)[0]==0),
                                np.count_nonzero(gen_epoch_types.reshape(1,-1)[0]==1),
                                np.count_nonzero(gen_epoch_types.reshape(1,-1)[0]==2),
                                np.count_nonzero(gen_epoch_types.reshape(1,-1)[0]==3)
    ]

    ax1.bar(epoch_types, gen_epoch_types_ditribution)

    # Epoch sequence
    ax2 = plt.subplot(row1[1])

    for i in range(len(epoch_types)):

        ordered_epochs_type_number = np.where(gen_epoch_types.reshape(1,-1)[0]==i, np.arange(num_epoch), np.nan)

        x_barh = np.transpose([ordered_epochs_type_number,np.ones(num_epoch)])
        y_barh = [i-0.25,0.5]

        ax2.broken_barh(x_barh, y_barh)

    ax2.set_yticks(np.arange(len(epoch_types)), epoch_types)

    # Display figure
    plt.show()


# Get the positions
# positions = np.array(session_data['positions'])
# times_videoFrames = np.array(session_data['timeofframes'])


# plot_trajectory_on_maze(arena_coordinates, tower_coordinates, all_trapezes_coordinates, reward_spouts_coordinates,
#                             xpositions, ypositions, time_video_frames, chunk_start=None, chunk_end=None,
#                             towerscolor=['k', 'k', 'k', 'k'], showtowerID=True, showspoutID=False, showdrops=True, show_arena_size=False, ax=None)


