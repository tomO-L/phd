from classes_and_functions import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import bisect

import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(call_pdb=False)


#################
### Functions ###
#################

# def is_rewarded(n_rewards, turns_since_last_reward) : #Function that sets up the kind of protocole you are using
    
#     # Here a turn takes 1 time unit to be performed

#     DELAY = 2
#     PLATEAU = 100
#     DEPLETING_SLOPE = 0.2

#     t = DEPLETING_SLOPE * (n_rewards - DELAY) # [DEPLETING_SLOPE] more turns than the previous one are requiered to get the reward
    
#     if n_rewards < DELAY :
        
#         t = 0
    
#     elif t > PLATEAU :

#         t = PLATEAU
    
#     result = 1 if turns_since_last_reward >= t else 0
    
#     return result

# def is_rewarded(X, since_last_reward) : #Function that sets up the kind of protocole you are using
    
#     #p is the probability to get a reward
    
#     DELAY = 2
#     PLATEAU = 100
#     DEPLETING_SLOPE = 0.2

#     if X < DELAY :
#         t = 0
    
#     else :
#         t = DEPLETING_SLOPE * (X - DELAY) # [DEPLETING_SLOPE] more turns than the previous one are requiered to get the reward
    
#     if t > PLATEAU :
#         t = PLATEAU
    
#     result = 1 if since_last_reward >= t else 0
    
#     # print(f"p = {p}, result = {result}")
#     return result

import random

def is_rewarded(X, since_last_reward) : #Function that sets up the kind of protocole you are using
    
    DELAY = 2
    PLATEAU = 100
    DEPLETING_SLOPE = 0.2

    #p is the probability to get a reward
    
    if X < DELAY :
        t = 0
    else :
        t = DEPLETING_SLOPE * (X - DELAY) # [DEPLETING_SLOPE] more turns than the previous one are requiered to get the reward
    if t > PLATEAU :
        t = PLATEAU
    p = 1 if since_last_reward >= t else 0
    result = 1 if random.random() <= p else 0
    # print(f"p = {p}, result = {result}")
    return result

def generate_reward_chronology(time):

    time_list =  np.arange(int(time))

    since_last_reward = 0

    reward_list = []

    tot_reward = 0

    for _ in time_list:

        reward = is_rewarded(tot_reward, since_last_reward)
        
        reward_list.append(reward)

        tot_reward += reward

        if reward:

            since_last_reward = 0

        else:

            since_last_reward += 1

    return np.cumsum(reward_list)

# def generate_reward_chronology(time):

#     n_tot_reward_list = [0]
#     n_turns_since_last_reward = 0
#     reward = 0

#     # Assuming every turn takes 1 time unit to be performed 
#     n_turns = int(time)


#     for i in range(n_turns+1):

#         # n_tot_reward = n_tot_reward_list[i]
#         n_tot_reward = n_tot_reward_list[-1]


#         reward = is_rewarded(n_tot_reward, n_turns_since_last_reward)

#         if reward:

#             n_turns_since_last_reward = 0
        
#         else:

#             n_turns_since_last_reward += 1

#         new_n_tot_reward = n_tot_reward + reward

#         n_tot_reward_list.append(new_n_tot_reward)

#     return n_tot_reward_list 

REWARD_CHRONOLOGY = generate_reward_chronology(200)
CHRONOLOGY = np.arange(len(REWARD_CHRONOLOGY))

def harvest_function(time,dt):

    # Assuming every turn takes 1 time unit to be performed 

    position_in_list = bisect.bisect(CHRONOLOGY,time)-1
    next_position_in_list = bisect.bisect(CHRONOLOGY,time+dt)-1

    # print(time,position_in_list,next_position_in_list, REWARD_CHRONOLOGY[position_in_list],REWARD_CHRONOLOGY[next_position_in_list])

    d_reward = REWARD_CHRONOLOGY[next_position_in_list] - REWARD_CHRONOLOGY[position_in_list]

    return d_reward



def exploitation_effort_function(time, dt):

    return 0

def travel_effort_function(time):

    return 0

def perform_foraging_session(t_travel, t_leave):

    agent_a = agent(t_travel, travel_effort_function)

    # patch_0 = patch(harvest_function, exploitation_effort_function)
    # patch_1 = patch(harvest_function, exploitation_effort_function)
    # patch_2 = patch(harvest_function, exploitation_effort_function)

    patch_0 = patch(harvest_function, exploitation_effort_function)
    # patch_1 = patch(harvest_function_2, exploitation_effort_function)


    # patches = [patch_0,patch_1,patch_2]
    patches = [patch_0]

    i = 0

    time_list = []
    energy_list = []

    while i<len(patches):

        agent_a.exploit_patch(patches[i])

        time_list.append(agent_a.time)
        energy_list.append(agent_a.energy)

        if patches[i].residence_time>t_leave:

            i += 1
            agent_a.travel()

    average_intake_rate = agent_a.energy/agent_a.time

    return time_list, energy_list, average_intake_rate

##################
### Parameters ###
##################

t_travel = 1
t_leave = 1

#############
### Plots ###
#############

# Initialisation
fig=plt.figure(figsize=(8, 4), dpi=300, constrained_layout=False, facecolor='w')

gs = fig.add_gridspec(1, 1)
row1 = gs[0].subgridspec(1, 2, width_ratios=[1,0.1], wspace=0.5)
ax = plt.subplot(row1[0,0])
ax_bis = plt.subplot(row1[0,1])

x0,y0,y1 = perform_foraging_session(t_travel, t_leave)
x1 = np.linspace(0,max(x0)+t_travel)

line0, = ax.plot(x0,y0)
line1, = ax.plot(x1 - t_travel, y1*x1)
line2, = ax_bis.plot(0,y1, marker='+', markersize=10)

ax.set_xlabel('Time')
ax.set_ylabel('Cummulated Energy')

ax_bis.set_xticks([])
ax_bis.grid()

fig.subplots_adjust(left=0.25, bottom=0.25)

# Sliders

ax_travel = fig.add_axes([0.25, 0.1, 0.65, 0.03])
travel_slider = Slider(
    ax=ax_travel,
    label='Travel Time',
    valmin=0.01,
    valmax=5,
    valinit=t_travel,
)

ax_leave = fig.add_axes([0.25, 0.05, 0.65, 0.03])
leave_slider = Slider(
    ax=ax_leave,
    label='Leaving Time',
    valmin=0.01,
    valmax=200,
    valinit=t_leave,
)

def update(val):

    x0, y0, y1 = perform_foraging_session(travel_slider.val, leave_slider.val)
    x1 = np.linspace(0,max(x0)+travel_slider.val)
    
    line0.set_xdata(x0)
    line0.set_ydata(y0)

    line1.set_xdata(x1 - travel_slider.val)
    line1.set_ydata(y1 * x1)

    line2.set_ydata([y1])

    ax_bis.set_ylim([0,1])
    ax_bis.set_xticks([])

    fig.canvas.draw_idle()

    print(x0,y0)

# ax.plot(CHRONOLOGY, REWARD_CHRONOLOGY)

travel_slider.on_changed(update)
leave_slider.on_changed(update)

plt.show()



# def reward_function(X, since_last_reward) : #Function that sets up the kind of protocole you are using
    
#     #p is the probability to get a reward
    
#     DELAY = 2
#     PLATEAU = 100
#     DEPLETING_SLOPE = 0.2

#     if X < DELAY :
#         t = 0
    
#     else :
#         t = DEPLETING_SLOPE * (X - DELAY) # [DEPLETING_SLOPE] more turns than the previous one are requiered to get the reward
    
#     if t > PLATEAU :
#         t = PLATEAU
    
#     p = 1 if since_last_reward >= t else 0
#     result = True if random.random() <= p else False
    
#     # print(f"p = {p}, result = {result}")
#     return result
    
#     # if random.random() <= p : return 1 #Sees if the random value is a greater value than the current threshold, given by the first part and a degressive function
#     # else : return 0



