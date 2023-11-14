import torch
import math
from DQN import hDQN
import os
import torch.nn as nn

class MetaController:

    def __init__(self, trained_meta_controller_weights_path):
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device('cpu')
        self.model_num = len([m for m in os.listdir(trained_meta_controller_weights_path) if m.startswith('meta_controller')])
        self.policy_nets = nn.ModuleList() #hDQN().to(self.device)
        # self.target_nets = nn.ModuleList() #hDQN().to(self.device)
        self.weights_path = trained_meta_controller_weights_path
        self.load_target_net_from_memory()

    def load_target_net_from_memory(self):

        for m in range(self.model_num):
            model_path = torch.load(os.path.join(self.weights_path,
                                                 'meta_controller_model_{0}.pt'.format(m)),
                                    map_location=self.device)
            policy_net = hDQN().to(self.device)
            policy_net.load_state_dict(model_path)
            self.policy_nets.append(policy_net)

    def get_goal_map(self, environment, agent):
        goal_map = torch.zeros_like(environment.env_map[:, 0, :, :])
        with torch.no_grad():
            env_map = environment.env_map.clone().to(self.device)
            need = agent.need.to(self.device)
            mean_output_values = torch.zeros_like(goal_map)
            rotation_num = 4  # including 0
            for m in range(self.model_num):
                for rot in range(rotation_num):
                    rotated_env = torch.rot90(env_map, k=rot, dims=[2, 3])
                    rotated_output_values = self.policy_nets[m](rotated_env, need)
                    output_values = torch.rot90(rotated_output_values, k=-rot, dims=[1, 2])
                    object_mask = environment.env_map.sum(dim=1)  # Either the agent or an object exists
                    output_values[object_mask < 1] = -math.inf
                    mean_output_values += output_values

            # for m in range(self.model_num): # original env
            #     for rot in range(3):
            #         output_values = self.policy_nets[m](torch.rot90()env_map, need)
            #         object_mask = environment.env_map.sum(dim=1)  # Either the agent or an object exists
            #         output_values[object_mask < 1] = -math.inf
            #         mean_output_values += output_values

            mean_output_values = mean_output_values / (self.model_num * rotation_num)
            goal_location = torch.where(torch.eq(mean_output_values, mean_output_values.max()))
            goal_location = torch.as_tensor([ll[0] for ll in goal_location][1:])

        goal_map[0, goal_location[0], goal_location[1]] = 1
        return goal_map, goal_location.unsqueeze(dim=0)
