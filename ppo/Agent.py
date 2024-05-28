import torch
import torch.nn as nn
import torch.distributions.categorical as Categorical
import numpy as np
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(torch.tensor(x))

    def get_action(self, x, greedy=False):
        logits = self.actor(torch.tensor(x))
        distribution = Categorical.Categorical(logits=logits)
        action = distribution.sample() if not greedy else distribution.mode
        return action

    def get_action_and_value(self, x, action=None):
        logits = self.actor(torch.tensor(x))
        distr =Categorical.Categorical(logits=logits)
        if action is None:
            action = distr.sample()
        return action, distr.log_prob(action), distr.entropy(), self.critic(x)