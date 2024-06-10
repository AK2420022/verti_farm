import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class Critic(nn.Module):
    def __init__(self, env,hidden_dim, hidden_layers,init_w=0.05):
        super(Critic, self).__init__()
        # Create layers for q1
        layers_q1 = []
        layers_q1.append(nn.Linear(np.array(env.single_observation_space.shape).prod() + np.array(env.action_space.shape).prod(), hidden_dim))
        layers_q1.append(nn.SiLU())
        for i in range(1, hidden_layers):
            layers_q1.append(nn.Linear(hidden_dim, hidden_dim))
            layers_q1.append(nn.SiLU())
        layers_q1.append(nn.Linear(hidden_dim, 1))
        self.q1 = nn.Sequential(*layers_q1)
        
        # Create layers for q2
        layers_q2 = []
        layers_q2.append(nn.Linear(np.array(env.single_observation_space.shape).prod() + np.array(env.action_space.shape).prod(), hidden_dim))
        layers_q2.append(nn.SiLU())
        for i in range(1, hidden_layers):
            layers_q2.append(nn.Linear(hidden_dim, hidden_dim))
            layers_q2.append(nn.SiLU())
        layers_q2.append(nn.Linear(hidden_dim, 1))
        self.q2 = nn.Sequential(*layers_q2)
        
        # Initialize weights
        self.init_weights()
    def init_weights(self):
        """
        The `init_weights` function initializes the weights of linear layers in ensembles using a specified
        initialization method.
        """
        init_w = 0.001
        for layer in self.q1:
            if isinstance(layer, nn.Linear):
                layer.weight.data.uniform_(-init_w, init_w)
                layer.weight.data.uniform_(-init_w, init_w)
        for layer in self.q2:
            if isinstance(layer, nn.Linear):
                layer.weight.data.uniform_(-init_w, init_w)
                layer.weight.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        """
        This Python function takes a state and an action as input, outputs the q values
        
        :param state: The input state
        :param action: Input action
        :return: The predicated q values 
        """
        q = torch.cat([torch.tensor(state).to(torch.float32), torch.tensor(action).to(torch.float32)], dim=-1)
        return self.q1(q), self.q2(q)

    def q1_forward(self, state, action):
        """
        This Python function takes a state and an action as input, outputs the q values 
        based only on one critic network
        
        :param state: The input state
        :param action: Input action
        :return: The predicated q values 
        """
        q =   torch.cat([torch.tensor(state).to(torch.float32), torch.tensor(action).to(torch.float32)], 1)
        q_values = self.critics[0](q)
        return q_values