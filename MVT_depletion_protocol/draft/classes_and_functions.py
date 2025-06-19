import numpy as np

#######################################################
####################    Classes    ####################
#######################################################

dt=0.00001

class patch():

    def __init__(self, harvest_function, exploitation_effort_function):

        # Define how energy is harvested and consumed
        self.harvest_function = harvest_function
        self.exploitation_effort_function = exploitation_effort_function
        
        self.residence_time = 0
        self.total_harvested_energy = 0


    # Define the effects of harvesting this patch
    def exploit(self):

        self.residence_time += dt

        harvested_energy = self.harvest_function(self.residence_time, dt)

        self.total_harvested_energy += harvested_energy

        consumed_energy = self.exploitation_effort_function(self.residence_time, dt)

        return harvested_energy, consumed_energy


class agent():

    def __init__(self, travel_time, travel_effort_function):

        # Begin with 0 energy at time 0
        self.energy = 0
        self.time = 0
        self.intake_rate = 0

        # Define travel time and its cost
        self.travel_time = travel_time
        self.travel_effort_function = travel_effort_function

    # Define the energy harvesting from a patch
    def exploit_patch(self, patch):

        self.time += dt

        harvested_energy, consumed_energy = patch.exploit()
        
        self.energy = self.energy + harvested_energy - consumed_energy
        self.intake_rate = harvested_energy/dt

    # Define the travel between two patches
    def travel(self):

        self.time += self.travel_time
        self.energy -= self.travel_effort_function(self.travel_time)

#####################################################
##################    Functions    ##################
#####################################################

def perform_foraging_session(t_travel, t_leave, patches, travel_effort_function):

    agent_a = agent(t_travel, travel_effort_function)

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

def create_environment(patch_types):

    environment = []

    for i in range(len(patch_types)):

        harvest_function, exploitation_effort_function = patch_types[i]

        patch_to_add = patch(harvest_function, exploitation_effort_function)

        environment.append(patch_to_add)

    return environment

def opt_finder(t_travel, time_range, patch_types, travel_effort_function):

    max_average_intake_rate = 0

    for t in time_range:

        patches = create_environment(patch_types)

        _, _, average_intake_rate = perform_foraging_session(t_travel, t, patches, travel_effort_function)

        if average_intake_rate > max_average_intake_rate:
        
            max_average_intake_rate = average_intake_rate
            t_opt = t
    
    _, _, average_intake_rate = perform_foraging_session(t_travel, t, patches, travel_effort_function)

    if t_opt==time_range[0] or t_opt==time_range[-1]:

        print("WARNING: t_opt is a border of t_range. You might want to widen t_range. ")


    return max_average_intake_rate

def opt_finder_v2(t_travel, time_range, patch_types, travel_effort_function):

    max_average_intake_rate = 0

    for t in time_range:

        patches = create_environment(patch_types)

        _, _, average_intake_rate = perform_foraging_session(t_travel, t, patches, travel_effort_function)

        if average_intake_rate > max_average_intake_rate:
        
            max_average_intake_rate = average_intake_rate
            t_opt = t
    
    _, _, average_intake_rate = perform_foraging_session(t_travel, t, patches, travel_effort_function)

    if t_opt==time_range[0] or t_opt==time_range[-1]:

        print("WARNING: t_opt is a border of t_range. You might want to widen t_range. ")


    return max_average_intake_rate


def cumulated_energy(harvest_function, exploitation_effort_function, time, dt):

    cumulated_energy = []

    current_cumulated_energy = 0

    for t in time:

        current_cumulated_energy = current_cumulated_energy + harvest_function(t,dt) - exploitation_effort_function(t, dt)

        cumulated_energy.append(current_cumulated_energy)

    return cumulated_energy

### Plot functions ###

def plot_cumulated_energy(ax, patch_type, time, dt, label='Patch cumulated energy'):

    harvest_function, exploitation_effort_function = patch_type
        
    cum_e = cumulated_energy(harvest_function, exploitation_effort_function, time, dt)

    ax.plot(time, cum_e, label = label)