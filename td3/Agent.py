import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
device = torch.device("cpu")

class Actor(nn.Module):
    def __init__(self, env):
        super(Actor, self).__init__()
        self.action_dim = env.action_space.high
        self.l1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

    
    def forward(self, state):
        a = F.relu(self.l1(torch.tensor(state).to(torch.float32).to("cpu")))
        a = F.relu(self.l2(a))
        return torch.tensor(self.action_dim).to("cpu") * torch.tanh(self.l3(a))
    

#Q1_theta1
class Critic(nn.Module):
    def __init__(self, env, num_critics):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 400)
        self.l2 = nn.Linear(400 + 1, 300)
        self.l3 = nn.Linear(300, 1)
        
        self.critics = nn.ModuleList([
            nn.Sequential(
                nn.Linear(np.array(env.single_observation_space.shape).prod() + 1, 400),
                nn.ReLU(),
                nn.Linear(400, 300),
                nn.ReLU(),
                nn.Linear(300, 1)
            )
            for _ in range(num_critics)
        ])

    def forward(self, state, action):
        q =  torch.cat([torch.tensor(state).clone().detach().requires_grad_(True).to(torch.float32), action.to(torch.float32)], 1)
        q_values = [critic(q) for critic in self.critics]
        return q_values
    
    def q1_forward(self, state, action):
        q =  torch.cat([torch.tensor(state.clone().detach().requires_grad_(True)).to(torch.float32), action.to(torch.float32)], 1)
        q_values = self.critics[0](q)
        return q_values