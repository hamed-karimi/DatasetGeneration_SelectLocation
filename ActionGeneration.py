import os
from os.path import exists as pexists

import matplotlib.pyplot as plt
import numpy as np
import torch

from ObjectFactory import ObjectFactory
from Utilities import Utilities
from Visualizer import Visualizer


def agent_reached_goal(environment, goal_map):
    has_same_map = torch.logical_and(environment.env_map[0, 0, :, :], goal_map[0, :, :])
    if has_same_map.sum() > 0:
        return True
    return False


def update_pre_located_objects(object_locations, agent_location, goal_reached):
    pre_located_objects = []
    prohibited_objects = []
    # if goal_reached:
    for obj_type in object_locations:
        temp1 = []
        for loc in obj_type:
            if any(~torch.eq(loc, agent_location[0])):
                temp1.append(loc.tolist())
            else:
                temp1.append([-1, -1])
                if any(~torch.eq(loc, torch.tensor([-1, -1]))):
                    prohibited_objects = loc.tolist()
        pre_located_objects.append(temp1)
    # prohibited_objects = [[]] if not any(prohibited_objects) else prohibited_objects
    return torch.tensor(pre_located_objects), torch.tensor(prohibited_objects)


def create_tensors(params):
    environments = torch.zeros((params.EPISODE_NUM,
                                params.STEPS_NUM,
                                params.OBJECT_TYPE_NUM + 1,
                                params.HEIGHT,
                                params.WIDTH), dtype=torch.float32)
    needs = torch.zeros((params.EPISODE_NUM,
                         params.STEPS_NUM,
                         params.OBJECT_TYPE_NUM), dtype=torch.float32)
    actions = torch.zeros((params.EPISODE_NUM,
                           params.STEPS_NUM), dtype=torch.int32)
    selected_goals = torch.zeros((params.EPISODE_NUM,
                                  params.STEPS_NUM,
                                  params.HEIGHT,
                                  params.WIDTH), dtype=torch.int32)
    goal_reached = torch.zeros((params.EPISODE_NUM,
                                params.STEPS_NUM), dtype=torch.bool)

    return environments, needs, actions, selected_goals, goal_reached


def generate_action():
    if not pexists('./Data'):
        os.mkdir('./Data')

    utility = Utilities()
    params = utility.get_params()
    factory = ObjectFactory(utility)
    res_folder = utility.make_res_folder()

    environment_initialization_prob_map = np.ones(params.HEIGHT * params.WIDTH) * 100 / (params.HEIGHT * params.WIDTH)
    meta_controller = factory.get_meta_controller()
    controller = factory.get_controller()

    print_threshold = 10
    visualizer = Visualizer(utility)
    environments, needs, actions, selected_goals, goal_reached = create_tensors(params)
    for episode in range(params.EPISODE_NUM):
        batch_environments_ll = []
        batch_actions_ll = []
        batch_needs_ll = []
        batch_selected_goal_maps_ll = []
        pre_located_objects_location = [[[]]] * params.OBJECT_TYPE_NUM
        pre_located_objects_num = torch.zeros((params.OBJECT_TYPE_NUM,), dtype=torch.int32)
        prohibited_object_locations = []
        object_amount_options = ['few', 'many']
        episode_object_amount = [np.random.choice(object_amount_options) for _ in range(params.OBJECT_TYPE_NUM)]

        agent = factory.get_agent(pre_location=[[]],
                                  preassigned_needs=[[]])
        environment = factory.get_environment(episode_object_amount,
                                              environment_initialization_prob_map,
                                              pre_located_objects_num,
                                              pre_located_objects_location,
                                              prohibited_object_locations)
        n_step = 0
        n_goal = 0
        while True:
            goal_map, goal_location = meta_controller.get_goal_map(environment,
                                                                   agent)  # goal type is either 0 or 1
            n_goal += 1
            while True:
                batch_environments_ll.append(environment.env_map.clone())
                batch_needs_ll.append(agent.need.clone())
                batch_selected_goal_maps_ll.append(goal_map.cpu().clone())

                if (agent.need > 20).any(): # or episode < print_threshold:
                    fig, ax = visualizer.map_to_image(agent, environment)
                    fig.savefig('{0}/episode_{1}_goal_{2}_step_{3}.png'.format(res_folder, episode, n_goal, n_step))
                    plt.close()

                agent_goal_map = torch.stack([environment.env_map[:, 0, :, :], goal_map], dim=1)
                action_id = controller.get_action(agent_goal_map).clone()
                agent.take_action(environment, action_id)

                step_goal_reached = agent_reached_goal(environment, goal_map)
                goal_reached[episode, n_step] = step_goal_reached

                batch_actions_ll.append(action_id.clone())
                # all_actions += 1
                n_step += 1

                if step_goal_reached or n_step == params.STEPS_NUM:
                    if step_goal_reached:
                        pre_located_objects_location, prohibited_object_locations = update_pre_located_objects(
                            environment.object_locations,
                            agent.location,
                            goal_reached)
                        pre_located_objects_num = environment.each_type_object_num
                        # pre_located_agent = agent.location.tolist()
                        # pre_assigned_needs = agent.need.tolist()

                        environment = factory.get_environment(episode_object_amount,
                                                              environment_initialization_prob_map,
                                                              pre_located_objects_num,
                                                              pre_located_objects_location,
                                                              prohibited_object_locations)
                    break

            if n_step == params.STEPS_NUM:
                break

        environments[episode, :, :, :, :] = torch.cat(batch_environments_ll, dim=0)
        needs[episode, :, :] = torch.cat(batch_needs_ll, dim=0)
        selected_goals[episode, :, :, :] = torch.cat(batch_selected_goal_maps_ll, dim=0)
        actions[episode, :] = torch.cat(batch_actions_ll, dim=0)

        if episode % 100 == 0:
            print(episode)

    # Saving to memory
    torch.save(environments, './Data/environments.pt')
    torch.save(needs, './Data/needs.pt')
    torch.save(selected_goals, './Data/selected_goals.pt')
    torch.save(goal_reached, './Data/goal_reached.pt')
    torch.save(actions, './Data/actions.pt')
