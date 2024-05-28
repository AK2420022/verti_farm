import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

class Critic(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.l1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 400)
        self.l2 = nn.Linear(400 + np.array(env.action_space.shape).prod(), 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, state, action):
        q = F.relu(self.l1(torch.tensor(state).to(torch.float32)))
        q = F.relu(self.l2(torch.cat([q, torch.tensor(action).to(torch.float32)], 1)))
        return self.l3(q)
