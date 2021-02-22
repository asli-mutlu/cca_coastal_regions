from AdaptationModel import *
import sys
import numpy as np

# Model parameters
model_width = 30  # width of simulation area
model_height = 30  # height of simulation area
init_population = 180  # initial population in simulation area
steps = 50  # number of time steps to run model


def slr_adaptation(slr_scenario='h', network_structure='aspatial', connection_radius=6, connection_percentage=0.2):
    if network_structure == 'spatial':
        network_parameter = connection_radius
    elif network_structure == 'aspatial':
        network_parameter = connection_percentage
    else:
        print('Error: Choose spatial or aspatial social network')
        sys.exit(1)

    # slr acceleration values come from Hall et al. (2016)
    if slr_scenario == 'h':
        slr_acceleration = 0.000163
    elif slr_scenario == 'm':
        slr_acceleration = 0.000113
    elif slr_scenario == 'l':
        slr_acceleration = 0.000063
    else:
        print('Error: Choose slr acceleration scenario: l, m, h')
        sys.exit(1)

    agent_states = []  # list for storing model output

    # initialize model
    model = AdaptationModel(model_width, model_height, init_population, slr_acceleration, network_structure,
                            network_parameter, 0)
    model.set_init_connections()

    # save initial agent state variables in agent_states
    init_agent_states = []
    for j in range(model_width):
        init_agent_states.append([])
    for cell in model.grid.coord_iter():
        cell_content, x, y = cell
        if type(cell_content) == Household:
            # collect agent state variables
            init_agent_states[x].append((cell_content.flood_damage, cell_content.inundation,
                                        cell_content.adaptive_capacity, cell_content.resistance,
                                        cell_content.accommodation, cell_content.attachment, cell_content.p_action,
                                        cell_content.p_resist, cell_content.p_accommodate, cell_content.p_retreat,
                                        cell_content.relative_elevation, cell_content.retreated))
        else:
            init_agent_states[x].append(np.nan)
    agent_states.append(init_agent_states)

    model.datacollector.collect(model)  # collect initial model state variables

    # run simulation
    for i in range(steps):
        model.step()
        agent_states_now = []  # list for agent state variables at current time step
        for j in range(model_width):
            agent_states_now.append([])
        for cell in model.grid.coord_iter():
            cell_content, x, y = cell
            if type(cell_content) == Household:
                # collect agent state variables
                agent_states_now[x].append((cell_content.flood_damage, cell_content.inundation,
                                            cell_content.adaptive_capacity, cell_content.resistance,
                                            cell_content.accommodation, cell_content.attachment, cell_content.p_action,
                                            cell_content.p_resist, cell_content.p_accommodate, cell_content.p_retreat,
                                            cell_content.relative_elevation, cell_content.retreated))
            else:
                agent_states_now[x].append(np.nan)
        agent_states.append(agent_states_now)
    model_attributes = model.datacollector.get_model_vars_dataframe()  # store model level state variables in dataframe

    return agent_states, model_attributes
