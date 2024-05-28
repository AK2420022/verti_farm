import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class Critic(nn.Module):
    def weight_init(self,m):
        init_w = 0.01
        if isinstance(m, nn.Linear):
            m.weight.data.uniform_(-init_w, init_w)
            m.weight.data.uniform_(-init_w, init_w)
    def __init__(self, env,hidden_dim, hidden_layers,init_w=0.05):
        super(Critic, self).__init__()
        layers = []
        layers.append(nn.Linear(np.array(env.single_observation_space.shape).prod() + np.array(env.action_space.shape).prod() , hidden_dim))
        layers.append(nn.SiLU())
        # Add hidden layers
        for i in range(1,hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        
        # Add output layer
        layers.append(nn.Linear(hidden_dim, np.array(env.action_space.shape).prod()))
        
        # Create Sequential model
        self.q1 = nn.Sequential(*layers)
        self.q2 = nn.Sequential(*layers)
        #self.q1.apply(self.weight_init)
        #self.q2.apply(self.weight_init)
    def init_weights(self, init_method):
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
        #print(self.critics[0][0].weight)
        q = torch.cat([torch.tensor(state).to(torch.float32), torch.tensor(action).to(torch.float32)], dim=-1)
        return self.q1(q), self.q2(q)

    def q1_forward(self, state, action):
        q =   torch.cat([torch.tensor(state).to(torch.float32), torch.tensor(action).to(torch.float32)], 1)
        q_values = self.critics[0](q)
        return q_values