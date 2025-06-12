import numpy as np

dt=0.01

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

