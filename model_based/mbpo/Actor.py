import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SquashBijector:
    @staticmethod
    def forward(x):
        return torch.tanh(x)

    @staticmethod
    def inverse(y, high):
        intermediate_result = 0.5 * torch.log((1 + y) / (1 - y))
        return intermediate_result

    @staticmethod
    def log_prob(log_prob, x):
        log_prob -= torch.sum(torch.log(1 - torch.tanh(x)**2 + 1e-6), dim=-1, keepdim=True)
        return log_prob

class Actor(nn.Module):
    def __init__(self, env, hidden_dim, hidden_layers, log_std_min=-20, log_std_max=1):
        super(Actor, self).__init__()
        self.action_space_type = type(env.action_space)
        
        if not isinstance(env.action_space, gym.spaces.Discrete):
            self.low = env.single_action_space.low
            self.high = env.single_action_space.high
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        layers = []
        layers.append(nn.Linear(np.array(env.observation_space.shape).prod(), hidden_dim))
        layers.append(nn.SiLU())
        
        for _ in range(1, hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        
        layers.append(nn.Linear(hidden_dim, np.array(env.action_space.shape).prod() * 2))
        self.actor = nn.Sequential(*layers)
       
    def init_weights(self):
        for layer in self.actor:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
    
    def forward(self, state):
        if not torch.is_tensor(state):
            state = torch.tensor(state)
        
        if state.ndim > 2 and state.size(1) == 1:
            state = state.squeeze(1)
        state = torch.tensor(state).to(torch.float32).to(device)
        a = self.actor(state)
        mean, log_std = torch.chunk(a, 2, dim=-1)
        #log_std = self.log_std_max - F.softplus(self.log_std_max - log_std)
        #log_std = self.log_std_min + F.softplus(log_std - self.log_std_min)
        return mean, log_std

    def get_action(self, state, deterministic=False, epsilon=1e-6):
        
        mean, log_std = self.forward(state)
        std = log_std.exp()

        distribution = torch.distributions.Normal(mean, std)
        if deterministic:
            action_ = mean 
        else:
            action_ = distribution.rsample()

        action = SquashBijector.forward(action_)
        log_prob = (distribution.log_prob(action_) - torch.log(1 - action.pow(2) + epsilon)).sum(-1, keepdim=True)

        if not isinstance(self.action_space_type, gym.spaces.Discrete):
            action = action * torch.tensor(self.high).to(device)
        
        return action, log_prob