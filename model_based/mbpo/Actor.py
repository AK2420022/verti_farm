import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
class SquashBijector:
    @staticmethod
    def forward(x):
        return torch.tanh(x)

    @staticmethod
    def inverse(y, high):
        # Compute the intermediate result without clamping
        intermediate_result = 0.5 * torch.log((1 + y) / (1 - y))
        return intermediate_result

    @staticmethod
    def log_prob(log_prob, x):
        # Apply the log-probability correction for the squashing operation
        log_prob -= torch.sum(torch.log(1 - torch.tanh(x)**2 + 1e-6), dim=-1, keepdim=True)
        return log_prob

class Actor(nn.Module):
    def __init__(self, env, hidden_dim,hidden_layers, log_std_min=-20, log_std_max=2, init_w=0.005):
        super(Actor, self).__init__()
        #self.env = env
        self.action_space_type = type(env.action_space)
        if isinstance(self.action_space_type,gym.spaces.Discrete) == False:
            self.low = env.single_action_space.low
            self.high =env.single_action_space.high
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        layers = []
        layers.append(nn.Linear(np.array(env.observation_space.shape).prod(), hidden_dim))
        layers.append(nn.SiLU())
        # Add hidden layers
        for i in range(1, hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        
        # Add output layer
        layers.append(nn.Linear(hidden_dim, np.array(env.action_space.shape).prod() * 2))
        # Create Sequential model
        self.actor = nn.Sequential(*layers)
       
        #self.actor.apply(self.weight_init)
    def init_weights(self, init_method):
        init_w = 0.005
        for layer in self.actor:
            if isinstance(layer, nn.Linear):
                layer.weight.data.uniform_(-init_w, init_w)
                layer.weight.data.uniform_(-init_w, init_w)
    def forward(self, state):
        a = self.actor(torch.tensor(state).to(torch.float32))
        mean, log_std = torch.chunk(a, 2, dim=-1)
        #log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        log_std = self.log_std_max - F.softplus(self.log_std_max - log_std)
        log_std = self.log_std_min + F.softplus(log_std - self.log_std_min)
        return mean, log_std

    def get_action(self, state, deterministic=False, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        mean = mean#.detach()

        distribution = torch.distributions.Normal(mean, std)
        if deterministic:
            action_ = mean 
        else:
            action_ = distribution.rsample()

        # Apply squash bijector forward transformation
        action = SquashBijector.forward(action_ * torch.tensor(self.high))
        # If the action space is continuous, scale actions to match the range
        if not isinstance(self.action_space_type, gym.spaces.Discrete):
            action = action  #* torch.tensor(self.high)
        # Calculate log probability
        # Apply inverse transformation of squash bijector to calculate log_prob
        log_prob = distribution.log_prob(action_).sum(-1, keepdim=True)
        #print("Log probability, ", log_prob)
        # Apply correction term for tanh-squashing
        #correction = torch.log(torch.tensor(self.high ) - atan_action.pow(2) + epsilon).sum(-1, keepdim=True)
        #print("log, " ,action.pow(2) )
        #print("torch.tensor(self.high), " , torch.log(torch.tensor(self.high ** 2 )) )
        #log_prob -= correction
        log_prob =  SquashBijector.log_prob(log_prob, action_)
        #print((torch.tensor(self.high) - tanh_action.pow(2) + epsilon))
        return action, log_prob