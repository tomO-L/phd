#######################
### Import packages ###
#######################

import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

########################
### Define functions ###
########################

# Data manipulation #

def load_data(folder_path_mouse_to_analyse,session_index):

    """
    Load pickle data file

    Arguments:
        folder_path_mouse_to_analyse (str): path to mouse folder
        session_index (int): index of the session from which to load the pickle data file
    
    Returns:
        (list) data from pickle file 

    """

    # Get all session folders that start with 'MOU' and sort them
    sessions_to_analyse = sorted([name for name in os.listdir(folder_path_mouse_to_analyse)
                                  if os.path.isdir(os.path.join(folder_path_mouse_to_analyse, name))
                                  and name.startswith('MOU')])

    session_to_analyse = sessions_to_analyse[session_index]

    # Define the output pickle filename and its full path
    output_pickle_filename = f"{session_to_analyse}_basic_processing_output.pickle"        
    output_pickle_filepath = os.path.join(folder_path_mouse_to_analyse, session_to_analyse, output_pickle_filename)

    # Open and load the session data from the pickle file
    with open(output_pickle_filepath, 'rb') as file:
        session_data = pickle.load(file)

    return session_data

# Plot #

def plot_trajectory_on_maze(arena_coordinates, tower_coordinates, all_trapezes_coordinates, reward_spouts_coordinates,
                            xpositions, ypositions, time_video_frames, chunk_start=None, chunk_end=None,
                            towerscolor=['k', 'k', 'k', 'k'], showtowerID=True, showspoutID=False, showdrops=True, show_arena_size=False, ax=None):
    """
    Plots the trajectory of a mouse on a maze with various elements like towers, reward spouts, and trapezes.

    Arguments:
        arena_coordinates (list): Coordinates defining the perimeter of the arena.
        tower_coordinates (list): Dictionary with tower names as keys and their vertex coordinates as values.
        all_trapezes_coordinates (list): Dictionary with tower names as keys and their associated trapeze coordinates as values.
        reward_spouts_coordinates (list): Dictionary with tower names as keys and their associated reward spout coordinates as values.
        xpositions (list): X-coordinates of the mouse's trajectory.
        ypositions (list): Y-coordinates of the mouse's trajectory.
        time_video_frames (list): Time stamps corresponding to the video frames.
        chunk_start (float, optional): Start time for plotting a chunk of the trajectory.
        chunk_end (float, optional): End time for plotting a chunk of the trajectory.
        towerscolor (list, optional): List of colors for the towers.
        showtowerID (bool, optional): Boolean to show tower IDs.
        showspoutID (bool, optional): Boolean to show reward spout IDs.
        showdrops (bool, optional): Boolean to show reward drops.
        show_arena_size (bool, optional): Boolean to show the arena size indicator.
        ax (matplotlib.axes.Axes, optional): Matplotlib axis object for plotting.
    """
    
    # Create a new figure and axis if no axis is provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # Draw the arena perimeter
    arena_x, arena_y = zip(*arena_coordinates + [arena_coordinates[0]])
    ax.plot(arena_x, arena_y, 'grey', linewidth=1)

    # Plot each tower
    for i, (tower_name, vertices) in enumerate(tower_coordinates.items()):
        tower_x, tower_y = zip(*vertices + [vertices[0]])
        ax.fill(tower_x, tower_y, towerscolor[i], alpha=0.01)
        ax.plot(tower_x, tower_y, 'k-', linewidth=0.5)
        if showtowerID:
            centroid_x = sum(x for x, y in vertices) / 4
            centroid_y = sum(y for x, y in vertices) / 4
            ax.text(centroid_x, centroid_y, tower_name, color='k', fontsize=4, ha='center', va='center', fontweight='normal')

    # Plot reward spouts as big blue dots with optional wall face ID
    if showdrops:
        for tower, spouts in reward_spouts_coordinates.items():
            for wall, (x, y) in spouts.items():
                ax.scatter(x, y, color='blue', s=1)
                if showspoutID:
                    ax.text(x, y, wall, color='black', fontsize=12, ha='center', va='center', fontweight='bold')

    # Plot each square and trapeze with the same color for each tower
    for i, (tower, trapezes) in enumerate(all_trapezes_coordinates.items()):
        for trapeze, coordinates in trapezes.items():
            coordinates_copy = coordinates + [coordinates[0]]
            x_coords, y_coords = zip(*coordinates_copy)
            ax.fill(x_coords, y_coords, 'c', alpha=0.1)

    # Plot the entire trajectory of the mouse or just a chunk
    if chunk_start is None:
        chunk_start = time_video_frames[0]
    if chunk_end is None:
        chunk_end = time_video_frames[-1]

    start_idx = np.searchsorted(time_video_frames, chunk_start)
    end_idx = np.searchsorted(time_video_frames, chunk_end)

    ax.plot(xpositions[start_idx:end_idx], ypositions[start_idx:end_idx], label='Trajectory', color='k', linewidth=0.25)

    # Draw arena size indicator if enabled
    arena_min_x = min(x for x, y in arena_coordinates)
    arena_max_x = max(x for x, y in arena_coordinates)
    arena_min_y = min(y for x, y in arena_coordinates)
    arena_max_y = max(y for x, y in arena_coordinates)

    arrow_y = arena_min_y - 0.05 * (arena_max_y - arena_min_y)
    if show_arena_size:
        ax.annotate('', xy=(arena_min_x-2, arrow_y), xytext=(arena_max_x+2, arrow_y),
                    arrowprops=dict(arrowstyle='<->', linewidth=0.5))
        ax.text((arena_min_x + arena_max_x) / 2, arrow_y - 2, '84 cm', ha='center', va='top', fontsize=5, fontweight='normal')

    # Set plot limits and remove axes/grid
    ax.set_xlim(arena_min_x-(arena_max_x-arena_min_x)*0.05, arena_max_x+(arena_max_x-arena_min_x)*0.05)
    ax.set_ylim(arrow_y - (arena_max_y-arena_min_y)*0.05, arena_max_y + (arena_max_y-arena_min_y)*0.05)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

def order_epochs(all_epochs):

    """
    Sort epochs in chronological order, omitting immobility epochs.

    Arguments:
        all_epoch (dict): dictionnary containing all the epochs, sorted by type of epoch. Each key is a different type of epoch.

    Returns:
        list: ordered_all_runs containing all the epochs omitting immobility epochs, sorted in chronological way.
        list: ordered_all_runs_frames containing all the epoch's frame intervals, omitting immobility epochs, sorted in chronological way.

    """

    # Initialize empty lists to store ordered runs and their first frames
    ordered_all_epochs = []
    ordered_all_epochs_frames = []

    # Loop through each key in the all_epochs dictionary
    for k in all_epochs.keys():
        
        # Loop through each run in the current key's list
        for i in range(len(all_epochs[k])):
            
            # Treats immobility differently because of a structure difference of the epoch variables
            if k != 'immobility':
        
                # Add the current run to ordered_all_runs
                ordered_all_epochs.append(all_epochs[k][i])
                # Add the first frame of the current run to ordered_all_runs_frames
                ordered_all_epochs_frames.append(all_epochs[k][i][0])

            else: 
            
                # TODO: deal with this hellish format

                start_frame = all_epochs[k][i][0]
                end_frame = all_epochs[k][i][1]

                reformated_epoch = all_epochs[k][i].copy()
                reformated_epoch.remove(reformated_epoch[0])
                reformated_epoch[0] = [start_frame,end_frame]

                # Add the current run to ordered_all_runs                
                ordered_all_epochs.append(reformated_epoch)
                # Add the first frame of the current run to ordered_all_runs_frames
                ordered_all_epochs_frames.append(reformated_epoch[0])

    # Sort the frames list based on the first element of each frame
    ordered_all_epochs_frames = sorted(ordered_all_epochs_frames, key=lambda x: x[0])
    # Sort the runs list based on the first element of each run
    ordered_all_epochs = sorted(ordered_all_epochs, key=lambda x: x[0])

    # Return the ordered lists of runs and their first frames
    return ordered_all_epochs, ordered_all_epochs_frames
