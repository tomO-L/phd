
import os
import pickle
import numpy as np
import pandas as pd
import ast
import matplotlib.path as mpath
from general_functions import *

def find_rbt_start_towerntrap(rbt):

    start = rbt[1]

    return start

def find_rbt_end_towerntrap(rbt):

    end = rbt[2]

    return end

def is_ext(loc):

    res = loc[1] in loc[0]

    return res
        
def compute_ext_start_trap(rbts):

    n_trap = 0

    for rbt in rbts:

        loc = find_rbt_start_towerntrap(rbt)

        if is_ext(loc):

            n_trap += 1

    return n_trap

def compute_int_start_trap(rbts):

    n_trap = 0

    for rbt in rbts:

        loc = find_rbt_start_towerntrap(rbt)

        if not(is_ext(loc)):

            n_trap += 1

    return n_trap

def compute_rbt_extremities(rbts):

    extremities_list = []

    for rbt in rbts:

        start_loc = find_rbt_start_towerntrap(rbt)
        end_loc = find_rbt_end_towerntrap(rbt)

        start_type = 'ext' if is_ext(start_loc) else 'int'
        end_type = 'ext' if is_ext(end_loc) else 'int'

        extremities_list.append(start_type + end_type)

    return extremities_list



def compute_ext_end_trap(rbts):

    n_trap = 0

    for rbt in rbts:

        loc = find_rbt_end_towerntrap(rbt)

        if is_ext(loc):

            n_trap += 1

    return n_trap

def compute_int_end_trap(rbts):

    n_trap = 0

    for rbt in rbts:

        loc = find_rbt_end_towerntrap(rbt)

        if not(is_ext(loc)):

            n_trap += 1

    return n_trap


def calculate_time_in_trapezes(folder_path_mouse_to_analyse, session_to_analyse, trapeze_width, time_start=None, time_end=None):
    
    """This function is used to calculate the total time spent in border, inner and trapezes areas during session. 
    Arguments:
        folder_path_mouse_to_analyse (string): path to the folder containing mouse's sessions folders
        session_to_analyse (string): session from which to compute the time and distance difference
        border_zone (dict): dictionnary of coordinates that define the border zone
        trapeze_width (float): value in pixel to set up the size of the trapezes
        time_start (float, optional): Time of the at which the trajectory will start being displayed
        time_end (float, optional): Time of the at which the trajectory will stop being displayed
                
    Outputs: 
        time_spent_in_zones (dict): dictionnary with area as key and time spent inside as value
    """

    # Build the path to the session pickle file
    pickle_file_path = os.path.join(folder_path_mouse_to_analyse, session_to_analyse, session_to_analyse + '_basic_processing_output.pickle')
    
    # Extract trapeze and tower data
    towers_coordinates = get_trapeze_and_tower_data(folder_path_mouse_to_analyse, session_to_analyse)[1]
    
    # Load data from pickle file
    with open(pickle_file_path, 'rb') as f:
        session_data = pickle.load(f)
    
    # Extract the necessary data from the pickle in the selected period
    smoothed_xpositions_cm = session_data['positions'][0] # x coordinates in cm
    smoothed_ypositions_cm = session_data['positions'][1] # y coordinates in cm
    times_videoFrames = session_data['timeofframes'] # Time of points in cm


    all_trapezes_coordinates_cm = generate_trapeze_and_tower_coordinates(towers_coordinates, trapeze_width)[0]

    # Chosing session's period to analyse
    if time_start is None:
        time_start = times_videoFrames[0]
    if time_end is None:
        time_end = times_videoFrames[-1]

    # Find the indices corresponding to the specified time interval
    start_idx = np.searchsorted(times_videoFrames, time_start)
    end_idx = np.searchsorted(times_videoFrames, time_end)

    # Extract data from the selected period
    selected_xpositions = smoothed_xpositions_cm[start_idx:end_idx]
    selected_ypositions = smoothed_ypositions_cm[start_idx:end_idx]
    selected_time = times_videoFrames[start_idx:end_idx]
    
    # Trapezes zones
    trapezes_polygons_by_tower = {
        'NW':None,
        'NE':None,
        'SW':None,
        'SE':None
    }

    for tower_key in all_trapezes_coordinates_cm.keys():

        trapezes = all_trapezes_coordinates_cm[tower_key]

        trapezes_polygons = []

        for trapeze, coords in trapezes.items():
            trapezes_polygons.append(mpath.Path(coords))
        
        trapezes_polygons_by_tower[tower_key] = trapezes_polygons


    # for tower, trapezes in all_trapezes_coordinates_cm.items():
    #     for trapeze, coords in trapezes.items():
    #         trapeze_polygons.append(mpath.Path(coords))
            
    # Variables to store time and distance spent in each zone
    time_in_trapeze = {
        'NW': {'N':0, 'E':0, 'S':0, 'W':0 },
        'NE': {'N':0, 'E':0, 'S':0, 'W':0 },
        'SW': {'N':0, 'E':0, 'S':0, 'W':0 },
        'SE': {'N':0, 'E':0, 'S':0, 'W':0 }
    }

    # Iterate on the trajectory positions and calculate the time and distance in each area
    for i in range(len(selected_xpositions) - 1):
        
        point_start = np.array([selected_xpositions[i], selected_ypositions[i]])

        # Calculate the time interval between each point
        dt = selected_time[i+1] - selected_time[i]

        # Test if the initial point is in the 'trapeze' (white) area

        for tower_key in trapezes_polygons_by_tower.keys():

            for i in range(len(trapezes_polygons_by_tower[tower_key])):

                if trapezes_polygons_by_tower[tower_key][i].contains_point(point_start):

                    trapeze_idx = list(time_in_trapeze[tower_key].keys())[i]

                    time_in_trapeze[tower_key][trapeze_idx] += dt # Horrible ptn

                    continue    
    
    return time_in_trapeze


def calculate_time_int_ext(time_in_tower, tower_id):

    ext_time = 0
    int_time = 0

    for k_trapeze in time_in_tower[tower_id].keys():

        if is_ext([tower_id,k_trapeze]):

            ext_time += time_in_tower[tower_id][k_trapeze]

        else:

            int_time += time_in_tower[tower_id][k_trapeze]

    return [int_time, ext_time]









