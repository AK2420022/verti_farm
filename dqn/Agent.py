import torch
import torch.nn as nn
import numpy as np
import copy

class Agent(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.q_network = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.single_action_space.n),
        )

        self.target_network = copy.deepcopy(self.q_network)

    def get_q_values(self, x):
        x = torch.tensor(x).to(torch.float32).to("cpu")
        return self.q_network(x).to("cpu")

    def get_target_values(self, x):
        x = torch.tensor(x).to(torch.float32).to("cpu")
        return self.target_network(x).to("cpu")

    def synchronize_networks(self):
        
        self.target_network.load_state_dict(self.q_network.state_dict())

    def get_action(self, x, greedy=True):
        x = torch.tensor(x).to(torch.float32).to("cpu")
        q_vals = self.get_q_values(x)
        action =None
        if len(q_vals.shape) > 1:
            action = q_vals.argmax(dim=1)
        else:
            action = q_vals.argmax()
        return action