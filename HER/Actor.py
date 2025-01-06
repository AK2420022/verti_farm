import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

class Actor(nn.Module):
    def __init__(self, env,init_w=0.1):
        super().__init__()
        """
        This Python function initializes a neural network with multiple linear layers for reinforcement
        learning tasks.
        
        :param env: The environment to initialize
        :param init_w: The weights to initialize
        """
        self.action_dim = env.action_space.high
        self.l1 = nn.Linear(np.array(env.observation_space["observation"].shape).prod(), 256)
        self.l1.weight.data.uniform_(-0.09, 0.05)
        self.l1.bias.data.uniform_(-0.09, 0.05)

        self.l2 = nn.Linear(256, 256)
        self.l2.weight.data.uniform_(init_w, -init_w)
        self.l2.bias.data.uniform_(init_w, -init_w)
        self.bn1 = nn.BatchNorm1d(256)

        self.l3 = nn.Linear(256, 128)
        self.l3.weight.data.uniform_(init_w, -init_w)
        self.l3.bias.data.uniform_(init_w, -init_w)
        self.bn2 = nn.BatchNorm1d(128)

        self.l5 = nn.Linear(128, np.array(env.action_space.shape).prod())
        self.l5.weight.data.uniform_(init_w, -init_w)
        self.l5.bias.data.uniform_(init_w, -init_w)
        self.bn4 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(p=0.5)
    def forward(self, state):
        q = F.relu(self.l1(torch.tensor(state).to(torch.float32).to("cpu")))
        q = self.dropout(F.relu((self.l2(q))))
        q =self.dropout(F.relu((self.l3(q))))
        #print(torch.tanh(self.l5(q)))
        return torch.tensor(self.action_dim).to("cpu") *  torch.tanh(self.l5(q))
