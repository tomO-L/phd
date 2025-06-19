import numpy as np

#######################################################
####################    Classes    ####################
#######################################################

dt=0.001

class patch():

    def __init__(self, harvest_function, exploitation_effort_function, dt=0.001):

        # Define how energy is harvested and consumed
        self.total_harvested_energy = 0
        self.residence_time = 0

        self.harvest_function = harvest_function
        self.exploitation_effort_function = exploitation_effort_function
        self.dt = dt
        

    # Define the effects of harvesting this patch
    def exploit(self):

        self.residence_time += dt

        harvested_energy = self.harvest_function(self.residence_time, dt)

        self.total_harvested_energy += harvested_energy

        consumed_energy = self.exploitation_effort_function(self.residence_time, dt)

        return harvested_energy, consumed_energy


class agent():

    def __init__(self, travel_time, travel_effort_function, dt=0.001):

        # Begin with 0 energy at time 0
        self.energy = 0
        self.time = 0
        self.intake_rate = 0

        # Define travel time and its cost
        self.dt = dt
        self.travel_time = travel_time
        self.travel_effort_function = travel_effort_function

    # Define the energy harvesting from a patch
    def exploit_patch(self, patch):

        self.time += dt

        harvested_energy, consumed_energy = patch.exploit()
        
        self.energy = self.energy + harvested_energy - consumed_energy
        self.intake_rate = (harvested_energy-consumed_energy)/dt

    # Define the travel between two patches
    def travel(self):

        self.time += self.travel_time
        self.energy -= self.travel_effort_function(self.travel_time)

#####################################################
##################    Functions    ##################
#####################################################

def perform_foraging_session(t_travel, rate_threshold, patches, travel_effort_function):

    agent_a = agent(t_travel, travel_effort_function)

    i = 0

    time_list = []
    energy_list = []
    leaving_time_perpatch = []

    while i<len(patches):

        agent_a.exploit_patch(patches[i])

        time_list.append(agent_a.time)
        energy_list.append(agent_a.energy)

        # instant_intake_rate = patches[i].total_harvested_energy/patches[i].residence_time
        instant_intake_rate = agent_a.intake_rate

        if instant_intake_rate <= rate_threshold:
        # if patches[i].residence_time>t_leave:

            leaving_time_perpatch.append(patches[i].residence_time)

            i += 1
            agent_a.travel()

    average_intake_rate = agent_a.energy/agent_a.time

    return time_list, energy_list, average_intake_rate, leaving_time_perpatch

def create_environment(patch_types):

    environment = []

    for i in range(len(patch_types)):

        harvest_function, exploitation_effort_function = patch_types[i]

        patch_to_add = patch(harvest_function, exploitation_effort_function)

        environment.append(patch_to_add)

    return environment


def cumulated_energy(harvest_function, exploitation_effort_function, time, dt):

    cumulated_energy = []

    current_cumulated_energy = 0

    for t in time:

        current_cumulated_energy = current_cumulated_energy + harvest_function(t,dt) - exploitation_effort_function(t, dt)

        cumulated_energy.append(current_cumulated_energy)

    return cumulated_energy

def compute_instant_intake_rate(harvest_function, exploitation_effort_function, last_time, dt):

    time = np.linspace(0,last_time, int(last_time/dt))

    cum_e = cumulated_energy(harvest_function, exploitation_effort_function, time, dt)

    res = (cum_e[-1] - cum_e[-2])/dt

    return res

def opt_finder_v2(t_travel, rate_threshold_list, patch_types, travel_effort_function,dt):


    close_enough = False

    while not(close_enough):

        dt = dt

        max_average_intake_rate = 0
        
        for rate_threshold in rate_threshold_list:


            patches = create_environment(patch_types)

            _, _, average_intake_rate, leaving_times = perform_foraging_session(t_travel, rate_threshold, patches, travel_effort_function)

            print("rate_threshold = ", rate_threshold, "average_intake_rate = ", average_intake_rate, "leaving_times = ", leaving_times)

            if average_intake_rate > max_average_intake_rate:
            
                max_average_intake_rate = average_intake_rate
                opt_leaving_times = leaving_times

        leaving_intake_rates = [compute_instant_intake_rate(patch_types[i][0], patch_types[i][1], opt_leaving_times[i], dt) for i in range(len(patch_types))]

        close_enough = (np.array(leaving_intake_rates) - max_average_intake_rate)**2 < 0.001

    if opt_leaving_times==rate_threshold_list[0] or opt_leaving_times==rate_threshold_list[-1]:

        print("WARNING: rate_threshold is a border of t_range. You might want to widen t_range. ")

    return max_average_intake_rate, opt_leaving_times



### Plot functions ###

# def plot_cumulated_energy(ax, patch_type, time, dt, label='Patch cumulated energy'):

#     harvest_function, exploitation_effort_function = patch_type
        
#     cum_e = cumulated_energy(harvest_function, exploitation_effort_function, time, dt)

#     ax.plot(time, cum_e, label = label)

def plot_cumulated_energy(ax, patch_type, time, dt, label='Patch cumulated energy'):

    harvest_function, exploitation_effort_function = patch_type
        
    # cum_e = 1-np.exp(-time) # cumulated_energy(harvest_function, exploitation_effort_function, time, dt)
    cum_e = cumulated_energy(harvest_function, exploitation_effort_function, time, dt)

    ax.plot(time, cum_e, label = label)
    