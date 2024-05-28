import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class EntropyNetwork(nn.Module):
    def __init__(self, env,hidden_dim, num_critics):
        super(EntropyNetwork, self).__init__()

        self.critics = nn.ModuleList([
            nn.Sequential(
                nn.Linear(np.array(env.single_observation_space.shape).prod() , hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim,1)
            )
            for _ in range(num_critics)
        ])

    def forward(self, state):
        q =  torch.tensor(state).to(torch.float32)
        q_values = [critic(q) for critic in self.critics]
        return q_values