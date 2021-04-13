from mesa import Agent, Model
from mesa.time import SimultaneousActivation
from mesa.space import SingleGrid
from mesa.datacollection import DataCollector
from matplotlib.colors import LinearSegmentedColormap
import random
import math

tick = 0  # counter for current time step


class AdaptationModel(Model):
    def __init__(self, width, height, n, slr_acceleration, network_structure, network_parameter, init_time=0):
        super().__init__()
        self.num_agents = n  # number of agents in simulation area
        self.width = width  # width of simulation grid
        self.height = height  # height of simulation grid
        self.slr_acceleration = slr_acceleration  # rate of mean sea level rise acceleration (m/yr^2)
        self.init_slr_increase = 0.007  # initial rate of mean sea level rise (m/yr) from Hall et al. (2016)
        self.network_mode = network_structure  # spatially dependent or spatially independent social network
        self.network_parameter = network_parameter  # social network density or radius
        self.schedule = SimultaneousActivation(self)  # agents are updated in a random order at each time step
        self.grid = SingleGrid(self.width, self.height, False)  # agents are located on a spatial grid
        self.sea_level = 0  # sea level relative to sea level at time=0
        self.storm_surge = 0  # height of maximum storm surge in each year
        self.community_resistance = 0  # level of resistance implemented at community level
        self.accommodation_incentive = 1  # community level scaling factor for probability of individual accommodation
        self.retreat_incentive = 1  # community level scaling factor for probability of individual retreat
        self.p_adapt = 0  # probability of community level adaptation
        self.agent_list = []  # list of all agents (retreated and not retreated)
        global tick
        tick = init_time

        # Create Agents -- agents are placed randomly on the grid and added to the model schedule
        locations = random.sample(range(self.width * self.height), self.num_agents)
        for i in range(self.num_agents):
            [x, y] = divmod(locations[i], self.height)
            a = Household(i, self, x, y)
            self.schedule.add(a)
            self.grid.place_agent(a, (a.x, a.y))
            self.agent_list.append(a)

        # Store model level attributes using a DataCollector object
        self.datacollector = DataCollector(
            model_reporters={"MSL": lambda m0: m0.sea_level, "Surge": lambda m1: m1.storm_surge,
                             "Population": lambda m2: m2.num_agents,
                             "Accommodation": lambda m4: m4.accommodation_incentive,
                             "Retreat": lambda m5: m5.retreat_incentive, "p_Adapt": lambda m6: m6.p_adapt}
        )

    def step(self):
        global tick
        self.calculate_inundation()  # calculate still water inundation level
        self.calculate_storm_surge()  # calculate maximum storm surge
        self.community_adaptation()  # implement community level adaptation actions
        self.schedule.step()  # update agents
        for a in self.agent_list:
            if a.retreated and a in self.schedule.agents:  # Remove retreated agents from model schedule and grid
                self.schedule.agents.remove(a)
                self.grid._remove_agent(a.pos, a)
                self.num_agents -= 1
        self.datacollector.collect(self)  # collect model level attributes for current time step
        tick += 1

    # Create social network. For aspatial network, possible connections form with a probability equal to the connection
    # percentage. For spatial network, connections form within specified radius
    def set_init_connections(self):
        if self.network_mode == 'aspatial':
            density = self.network_parameter
            for i in self.schedule.agents:
                for j in self.schedule.agents:
                    if j.unique_id > i.unique_id:
                        x = random.random()
                        if x < density:
                            i.connections.append(j)
                            j.connections.append(i)
        else:
            radius = self.network_parameter
            for i in self.schedule.agents:
                for j in self.schedule.agents:
                    if abs(i.x - j.x) < radius and abs(i.y - j.y) < radius and i.unique_id != j.unique_id:
                        i.connections.append(j)
                        j.connections.append(i)

    # Calculate still water inundation using the constant slr acceleration
    def calculate_inundation(self):
        self.sea_level = self.init_slr_increase*tick + self.slr_acceleration*tick**2

    # Calculate height of maximum annual storm surge from annual exceedance probability
    def calculate_storm_surge(self):
        aep = random.random()
        self.storm_surge = max(0, -math.log(2*aep)/1.164)

    # Implement community level adaptation
    # Probability of adaptation depends on the average flood damage and inundation levels
    # Resistance affects inundation and flooding for all agents. Accommodation and retreat implementation are considered
    # incentives for agents to implement actions on their own. In each year, there is a 20% chance that the
    # accommodation and retreat incentives will decrease by 1
    def community_adaptation(self):
        x = random.random()
        total_damage = 0
        total_inundation = 0
        for a in self.schedule.agents:
            total_damage += a.flood_damage
            total_inundation += a.inundation
        if self.num_agents > 0:
            average_damage = min(10, total_damage/self.num_agents)
            average_inundation = max(0, min(total_inundation/self.num_agents, 0.5))
            self.p_adapt = 0.09*average_damage + 1.8*average_inundation - 0.18*average_damage*average_inundation + 0.1
            if self.community_resistance < 3:
                if x < self.p_adapt * 0.6:
                    self.community_resistance += 1
                elif x < self.p_adapt * 0.3:
                    self.accommodation_incentive += 1
                elif x < self.p_adapt:
                    self.retreat_incentive += 1
            else:
                if x < self.p_adapt * 0.7:
                    self.accommodation_incentive += 1
                elif x < self.p_adapt:
                    self.retreat_incentive += 1
            if random.random() < 0.1:
                self.accommodation_incentive = max(1, self.accommodation_incentive-1)
            if random.random() < 0.1:
                self.retreat_incentive = max(1, self.retreat_incentive-1)


class Household(Agent):
    def __init__(self, unique_id, model, x, y):
        super().__init__(unique_id, model)
        self.unique_id = unique_id  # unique identification number
        self.x = x  # x-coordinate of location in grid
        self.y = y  # y-coordinate of location in grid
        self.connections = []  # list of all agents to which ego is connected
        self.elevation = self.x/self.model.width*10  # elevation above initial sea level
        self.adaptive_capacity = random.normalvariate(0, 1)  # ability to implement adaptation actions
        self.resistance = 0  # cumulative number of resistance actions undertaken
        self.accommodation = 0  # cumulative number of accommodation actions undertaken
        self.p_action = 0  # probability of implementing an adaptation action
        self.p_resist = 0.6  # probability that action implemented will be resistance
        self.p_accommodate = 0.3  # probability that action implemented will be accommodation
        self.p_retreat = 0.1  # probability that action implemented will be retreat
        self.flood_damage = 0  # flood damage remembered by ego
        self.attachment = 1  # attachment to place (proportion of connections that have not retreated)
        self.relative_elevation = self.elevation  # elevation above current sea level
        self.inundation = 0  # level of water above relative elevation plus implemented adaptation
        self.retreated = False  # retreated from simulation grid

    def step(self):
        self.calculate_adaptive_capacity()  # calculate new adaptative capacity
        self.calculate_attachment()  # calculate attachment to place
        self.apply_inundation()  # calculate relative elevation
        self.apply_damage()  # calculate current memory of damage from storm surge

    def advance(self):
        self.make_decision()  # implement adaptation actions

    # An agent's attachment to place is calculated as the proportion of their connections that have not retreated
    def calculate_attachment(self):
        retreated_connections = 0
        remaining_connections = 0
        for a in self.connections:
            if a.retreated:
                retreated_connections += 1
            else:
                remaining_connections += 1
        if retreated_connections + remaining_connections > 0:
            self.attachment = remaining_connections/(retreated_connections + remaining_connections)
        else:
            self.attachment = 0

    # Subtract current sea level from elevation to calculate relative elevation
    # Add resistance and accommodation to relative elevation to calculate inundation. Negative inundation implies
    # elevation above sea level.
    def apply_inundation(self):
        self.relative_elevation = self.elevation - self.model.sea_level
        resistance_level = max(self.resistance, self.model.community_resistance)
        self.inundation = -(self.relative_elevation + resistance_level + self.accommodation)

    # Flood damage is defined as the height of the annual maximum storm surge added to inundation
    # If accommodation measures have been implemented, flood damage is decreased 5% for each meter of accommodation
    # Memory of past flood damage is reduced by 10% each year
    def apply_damage(self):
        self.flood_damage *= 0.9
        if self.model.storm_surge > 0:
            flood_height = max(0, self.model.storm_surge + self.inundation)
            self.flood_damage = flood_height * (1 - self.accommodation * 0.05)

    # Adaptive capacity changes randomly each year
    def calculate_adaptive_capacity(self):
        self.adaptive_capacity += random.normalvariate(0, 0.1)

    # The probability that an agent implements an action depends on flood damage and relative elevation
    # The relative probabilities of each possible type of action depend on community level adaptions and attachment
    # to place. The first resistance or accommodation action effectively reduces inundation by 1 m. The second and
    # third resistance or accommodation actions reduce inundation by 0.5 m each. An agent cannot implement more than 2 m
    # of accommodation or resistance. Agents that retreat are removed from the grid.
    def make_decision(self):
        d = min(self.flood_damage, 10)
        i = min(1, max(self.inundation, 0))
        self.p_action = 0.09*d + 1.8*i - 0.18*d*i + 0.1
        if self.resistance >= 2 or self.adaptive_capacity <= 0:
            self.p_resist = 0
        if self.accommodation < 2 and self.adaptive_capacity > -0.1 * (self.model.accommodation_incentive - 1):
            self.p_accommodate *= self.model.accommodation_incentive
        else:
            self.p_accommodate = 0
        if self.adaptive_capacity > -0.1 * (self.model.retreat_incentive - 1):
            if self.attachment != 0:
                self.p_retreat = self.p_retreat * self.model.retreat_incentive / self.attachment
            else:
                self.p_retreat *= self.model.retreat_incentive
        else:
            self.p_retreat = 0
        x = random.uniform(0, 1)
        if self.p_resist + self.p_accommodate + self.p_retreat != 0:
            self.p_resist = self.p_resist / (self.p_resist+self.p_accommodate+self.p_retreat) * self.p_action
            self.p_accommodate = self.p_accommodate / (self.p_resist+self.p_accommodate+self.p_retreat) * self.p_action
            self.p_retreat = self.p_retreat / (self.p_resist+self.p_accommodate+self.p_retreat) * self.p_action
            if x < self.p_resist:
                if self.resistance == 0:
                    self.resistance = 1
                elif self.resistance == 1:
                    self.resistance = 1.5
                else:
                    self.resistance = 2
            elif x < (self.p_resist+self.p_accommodate):
                if self.accommodation == 0:
                    self.accommodation = 1
                elif self.accommodation == 1:
                    self.accommodation = 1.5
                else:
                    self.accommodation = 2
            elif x < self.p_action:
                for a in self.model.schedule.agents:
                    if a.unique_id == self.unique_id:
                        self.retreated = True

# define color maps for animation
cdict_damage = {'red': ((0.0, 0.7, 0.7), (1.0, 1.0, 1.0)),
                'green': ((0.0, 0.6, 0.6), (1.0, 0.1, 0.1)),
                'blue': ((0.0, 0.6, 0.6), (1.0, 0.1, 0.1))}
cm_damage = LinearSegmentedColormap('DamageColor', cdict_damage)
cdict_inundation = {'red': ((0.0, 0.4, 0.4), (0.5, 0.5, 0.5), (1.0, 0.2, 0.2)),
                    'green': ((0.0, 1.0, 1.0), (0.5, 0.5, 0.5), (1.0, 0.0, 0.0)),
                    'blue': ((0.0, 0.0, 0.0), (0.5, 0.5, 0.5), (1.0, 1.0, 1.0))}
cm_inundation = LinearSegmentedColormap('InundationColor', cdict_inundation)
cdict_ac = {'red': ((0.0, 1.0, 1.0), (0.5, 0.8, 0.8), (1.0, 0.0, 0.0)),
            'green': ((0.0, 0.0, 0.0), (0.5, 0.8, 0.8), (1.0, 1.0, 1.0)),
            'blue': ((0.0, 0.2, 0.2), (0.5, 0.8, 0.8), (1.0, 0.2, 0.2))}
cm_ac = LinearSegmentedColormap('InundationColor', cdict_ac)
cdict_adaptation = {'red': ((0.0, 0.8, 0.8), (1.0, 0.0, 0.0)),
                    'green': ((0.0, 0.8, 0.8), (1.0, 0.0, 0.0)),
                    'blue': ((0.0, 0.8, 0.8), (1.0, 0.0, 0.0))}
cm_adaptation = LinearSegmentedColormap('InundationColor', cdict_adaptation)
