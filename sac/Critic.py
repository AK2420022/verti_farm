import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class Critic2(nn.Module):
    def __init__(self, env,hidden_dim, num_critics):
        super(Critic2, self).__init__()
        self.l1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + np.array(env.action_space.shape).prod() , hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, np.array(env.action_space.shape).prod())

    def forward(self, state, action):
        #print(self.critics[0][0].weight)
        q =  torch.cat([torch.tensor(state).to(torch.float32), torch.tensor(action).to(torch.float32)], 1)
        a = F.relu(self.l1(q),inplace=True)
        a = F.relu(self.l2(a),inplace=True)
        return self.l3(a)
    
    def q1_forward(self, state, action):
        q =   torch.cat([torch.tensor(state).to(torch.float32), torch.tensor(action).to(torch.float32)], 1)
        q_values = self.critics[0](q)
        return q_values

class Critic1(nn.Module):
    def __init__(self, env,hidden_dim, num_critics):
        super(Critic1, self).__init__()
        self.l1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + np.array(env.action_space.shape).prod() , hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, np.array(env.action_space.shape).prod())

    def forward(self, state, action):
        #print(self.critics[0][0].weight)
        q =  torch.cat([torch.tensor(state).to(torch.float32), torch.tensor(action).to(torch.float32)], 1)
        a = F.relu(self.l1(q),inplace=True)
        a = F.relu(self.l2(a),inplace=True)
        return self.l3(a)
    
    def q1_forward(self, state, action):
        q =   torch.cat([torch.tensor(state).to(torch.float32), torch.tensor(action).to(torch.float32)], 1)
        q_values = self.critics[0](q)
        return q_values