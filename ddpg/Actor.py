import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

# This class defines an actor model for reinforcement learning with a neural network architecture that
# takes a state as input and outputs an action.
class Actor(nn.Module):

    def __init__(self, env):
        """
        This Python function initializes a neural network with multiple linear layers for reinforcement
        learning tasks.
        
        :param env: The environment to initialize
        """
        super().__init__()
        self.action_dim = env.action_space.high
        self.l1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, 64)
        self.l4 = nn.Linear(64, 64)
        self.l5 = nn.Linear(64, 1)

    
    def forward(self, state):
        """
        The `forward` function takes a state as input, passes it through several layers of a neural network
        with ReLU activation functions, and outputs an action scaled by the action dimension and passed
        through a hyperbolic tangent function.
        
        :param state: observation or state
        :return: action predicted by the actor
        """

        a = F.relu(self.l1(torch.tensor(state).to(torch.float32).to("cpu")))
        a = F.relu(self.l2(a))
        a = F.relu(self.l3(a))
        a = F.relu(self.l4(a))

        return torch.tensor(self.action_dim).to("cpu") * torch.tanh(self.l5(a))
