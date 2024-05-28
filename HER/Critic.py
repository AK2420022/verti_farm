import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

class Critic(nn.Module):
    def __init__(self, env,init_w = -0.1,end_w = 0.1):
        super().__init__()
        self.l1 = nn.Linear(np.array(env.observation_space["observation"].shape).prod()+np.array(env.action_space.shape).prod(), 256)
        self.l1.weight.data.uniform_( init_w,end_w)
        self.l1.bias.data.uniform_( init_w,end_w)

        self.l2 = nn.Linear(256, 256)
        self.l2.weight.data.uniform_( init_w,end_w)
        self.l2.bias.data.uniform_( init_w,end_w)
        self.bn1 = nn.BatchNorm1d(256)

        self.l3 = nn.Linear(256, 128)
        self.l3.weight.data.uniform_( init_w,end_w)
        self.l3.bias.data.uniform_( init_w,end_w)

        self.l4 = nn.Linear(128, 1)
        self.l4.weight.data.uniform_( init_w,end_w)
        self.l4.bias.data.uniform_( init_w,end_w)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.6)
    def forward(self, state, action):
        #print(self.l4.weight)
        q = F.relu(self.l1(torch.cat([torch.tensor(state), torch.tensor(action).to(torch.float32)],1).to(torch.float32)))
        
        q = self.dropout(F.relu(self.bn2(self.l2(q))))
        q = self.dropout(F.relu(self.bn3(self.l3(q))))
        return self.l4(q)
