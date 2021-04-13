from RunModel import *
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib
matplotlib.use('TkAgg')

# User specified inputs
# Choose agent slr scenario, type of social network, network parameter, and number of simulations
slr_scenario = 'h'  # h, m, or l
network_mode = 'aspatial'  # spatial or aspatial
connection_radius = 5  # radius of connection for spatial network
connection_percentage = 0.2  # density for aspatial network
simulations = 10  # number of simulations run

output = []  # list of model output from each simulation run
for sim in range(simulations):
    output.append(slr_adaptation(slr_scenario, network_mode, connection_radius, connection_percentage))  # run model

# Visualize output

# Animation: choose one agent attribute from one run to animate on a grid
# Attribute codes -- 0: flood damage, 1: inundation, 2: adaptive capacity, 3: resistance, 4: accommodation,
# 5: attachment, 6: p_action, 7: p_resist, 8: p_accommodate, 9: p_retreat or -1 for no animation
# view_simulation must be less than or equal to simulations
animated_attribute = 1  # agent attribute to animate
if animated_attribute >= 0:
    view_simulation = 0  # simulation run to animate
    fig_animation = plt.figure()
    agent_grids = []
    for t in range(steps):
        agent_map = np.empty((model_height, model_width))
        agent_map[:] = np.NAN
        this_year_output = output[view_simulation][0][t]
        for i in range(model_width):
            for j in range(model_height):
                if this_year_output[i][j] is not np.nan:
                    agent_map[j][i] = this_year_output[i][j][animated_attribute]
        agent_grids.append(agent_map)
    if animated_attribute == 0:
        color_map = cm_damage
        minval = 0
        maxval = 4
        title = 'Damage in Agent Memory (m)'
    elif animated_attribute == 1:
        color_map = cm_inundation
        minval = -2
        maxval = 2
        title = 'Level of Inundation (m)'
    elif animated_attribute == 2:
        color_map = cm_ac
        minval = -2
        maxval = 2
        title = 'Adaptive Capacity'
    elif animated_attribute == 3:
        color_map = cm_adaptation
        minval = 0
        maxval = 3
        title = 'Number of Resistance Actions Implemented'
    elif animated_attribute == 4:
        color_map = cm_adaptation
        minval = 0
        maxval = 3
        title = 'Number of Accommodation Actions Implemented'
    else:
        color_map = cm_adaptation
        minval = 0
        maxval = 1
        if animated_attribute == 5:
            title = 'Attachment'
        elif animated_attribute == 6:
            title = 'Probability of Implementing an Adaptation Action'
        elif animated_attribute == 7:
            title = 'Probability of Implementing an Resistance Action'
        elif animated_attribute == 8:
            title = 'Probability of Implementing an Accommodation Action'
        elif animated_attribute == 9:
            title = 'Probability of Retreating'
        else:
            print('Error: Choose attribute to animate: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9')
            sys.exit(1)
    im = plt.imshow(agent_grids[0], cmap=color_map, vmin=minval, vmax=maxval)

    # Animation functions
    def init():
        im.set_data(agent_grids[0])


    def animate(k):
        im.set_data(agent_grids[k])
        return im

    # create animation of selected attributed
    anim = animation.FuncAnimation(fig_animation, animate, init_func=init, frames=steps, repeat=False, interval=500)
    plt.colorbar()
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')


# Choose one agent attribute to average at each time step and plot as a time series for each run
# Attribute codes -- 0: flood damage, 1: inundation, 2: adaptive capacity, 5: attachment, 6: p_action, 7: p_resist,
# 8: p_accommodate, 9: p_retreat, 10: relative elevation
agent_attribute = 0
fig_agent_average = plt.figure()
for sim in range(simulations):
    average_value = []
    value_sum = 0
    count = 0
    for t in range(steps):
        value_output = output[sim][0][t]
        for i in range(model_width):
            for j in range(model_height):
                if value_output[i][j] is not np.nan:
                    value_sum += value_output[i][j][agent_attribute]
                    count += 1
        average_value.append(value_sum/count)
    plt.plot(average_value, label='run ' + str(sim))
plt.legend()
if agent_attribute == 0:
    plt.ylabel('Mean Flood Damage (m)')
elif agent_attribute == 1:
    plt.ylabel('Mean Inundation (m)')
elif agent_attribute == 2:
    plt.ylabel('Mean Adaptive Capacity')
elif agent_attribute == 5:
    plt.ylabel('Mean Attachment Factor')
elif agent_attribute == 6:
    plt.ylabel('Mean Probability of Implenting Any Adaptation Action')
elif agent_attribute == 7:
    plt.ylabel('Mean Probability of Implenting Resistance')
elif agent_attribute == 8:
    plt.ylabel('Mean Probability of Implenting Accommodation')
elif agent_attribute == 9:
    plt.ylabel('Mean Probability of Retreat')
elif agent_attribute == 10:
    plt.ylabel('Mean Relative Elevation (m)')
else:
    print('Choose attribute to plot mean value: 0, 1, 2, 5, 6, 7, 8, 9, 10')
plt.xlabel('Time (years)')

# plot population that has not retreated as a time series for each run
# plot percent reduction in population, and year to reach 95% of reduction
retreat_metrics = np.empty((simulations, 2))
fig_pop = plt.figure()
plt.ylabel('Population')
plt.xlabel('Time (years)')
for sim in range(simulations):
    population = output[sim][1].Population
    plt.plot(population, label='run ' + str(sim))
    # calculate number of agents that retreated, percent of agents that retreated, and time for 95% to retreat
    reduction = population[0] - population[steps]
    percent_decrease = reduction / population[0]
    reduction_time = steps
    for j in range(steps):
        if population[0] - population[j] > 0.95 * reduction:
            reduction_time = j
            break
    retreat_metrics[sim, :] = percent_decrease, reduction_time
plt.legend()
fig_metrics, reduction_ax = plt.subplots()
reduction_ax.plot(retreat_metrics[:, 0], 'b-')
reduction_ax.set_ylabel('Percent Population Decrease', color='b')
time_ax = reduction_ax.twinx()
time_ax.plot(retreat_metrics[:, 1], 'r-')
time_ax.set_ylabel('Years to 95% of Reduction', color='r')
reduction_ax.set_xlabel('Run Number')

# plot extreme water level as a time series for each run
fig_surge = plt.figure()
for sim in range(simulations):
    storm_surge = output[sim][1].Surge + output[sim][1].MSL
    plt.plot(storm_surge, label='run ' + str(sim))
plt.legend()
plt.ylabel('Extreme Water Level (m)')
plt.xlabel('Time (years)')
plt.show()
