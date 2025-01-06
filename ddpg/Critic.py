import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

# This class defines a critic neural network for reinforcement learning with two hidden layers.
class Critic(nn.Module):
    def __init__(self, env):
        """
        This Python function initializes neural network layers based on the environment's observation and
        action space dimensions.
        
        :param env: Environment to initialize
        """
        super().__init__()
        self.l1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 400)
        self.l2 = nn.Linear(400 + np.array(env.action_space.shape).prod(), 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, state, action):
        """
        The `forward` function takes a state and an action as input, passes the state through two linear
        layers with ReLU activation, concatenates the output with the action, and passes it through a final
        linear layer to produce the output.
        
        :param state: The `state` parameter in the `forward` method represents the current state of the
        environment or system in a reinforcement learning context.
        :return: The Q value of the combination of state and action
        """
        q = F.relu(self.l1(torch.tensor(state).to(torch.float32)))
        q = F.relu(self.l2(torch.cat([q, torch.tensor(action).to(torch.float32)], 1)))
        return self.l3(q)
