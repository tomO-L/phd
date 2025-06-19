from classes_and_functions import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider

import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(call_pdb=False)


#################
### Functions ###
#################

def is_rewarded(n_rewards, turns_since_last_reward) : #Function that sets up the kind of protocole you are using
    
    # Here a turn takes 1 time unit to be performed

    DELAY = 2
    PLATEAU = 100
    DEPLETING_SLOPE = 0.2

    t = DEPLETING_SLOPE * (n_rewards - DELAY) # [DEPLETING_SLOPE] more turns than the previous one are requiered to get the reward
    
    if n_rewards < DELAY :
        
        t = 0
    
    elif t > PLATEAU :

        t = PLATEAU
    
    result = 1 if turns_since_last_reward >= t else 0
    
    return result

def shitty_harvest_function(time, dt):

    n_tot_reward = 0
    n_turns_since_last_reward = 0

    n_turns = int(time)

    reward = 0

    for i in range(n_turns+1):

        # if i==n_turns-1:
        
        #     n_turns_since_last_reward = time

        reward = is_rewarded(n_tot_reward, n_turns_since_last_reward)

        if reward:

            n_turns_since_last_reward = 0
        
        else:

            n_turns_since_last_reward += 1

        n_tot_reward += reward

    print(reward)

    return reward*dt # Shitty fix here


# def harvest_function(time, dt):

#     tau = 1

#     d_energy = np.exp(-time/tau)*(1 - np.exp(-dt/tau))

#     return d_energy

def harvest_function_1(time, dt):

    tau = 1

    d_energy = np.exp(-time/tau)*(1 - np.exp(-dt/tau))

    return d_energy

def harvest_function_2(time, dt):

    tau = 2

    d_energy = np.exp(-time/tau)*(1 - np.exp(-dt/tau))

    return d_energy


def exploitation_effort_function(time, dt):

    return 0

def travel_effort_function(time):

    return 0

def perform_foraging_session(t_travel, t_leave):

    agent_a = agent(t_travel, travel_effort_function)

    # patch_0 = patch(harvest_function, exploitation_effort_function)
    # patch_1 = patch(harvest_function, exploitation_effort_function)
    # patch_2 = patch(harvest_function, exploitation_effort_function)

    patch_0 = patch(harvest_function_1, exploitation_effort_function)
    # patch_1 = patch(harvest_function_2, exploitation_effort_function)


    # patches = [patch_0,patch_1]
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
t_leave = 1.1433

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
    valmax=1,
    valinit=t_travel,
)

ax_leave = fig.add_axes([0.25, 0.05, 0.65, 0.03])
leave_slider = Slider(
    ax=ax_leave,
    label='Leaving Time',
    valmin=0.01,
    valmax=5,
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

travel_slider.on_changed(update)
leave_slider.on_changed(update)

plt.show()