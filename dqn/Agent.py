import torch
import torch.nn as nn
import numpy as np
import copy
# This class defines an Agent with Q-learning networks for making decisions in a reinforcement
# learning environment.

class Agent(nn.Module):
    def __init__(self, env):
        """
        The function initializes a neural network for Q-learning with a target network that is a deep copy
        of the Q-network.
        
        :param env:  The `env` parameter
        represent the environment in which the agent will interact
        """
        super().__init__()
        self.q_network = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.single_action_space.n),
        )

        self.target_network = copy.deepcopy(self.q_network)

    def get_q_values(self, x):
        """
        The function `get_q_values` takes input `x`, converts it to a torch tensor of type float32 on CPU,
        and returns the output of the q_network model also on CPU.
        
        :param x: input x
        :return: returns the output target values
        """
        x = torch.tensor(x).to(torch.float32).to("cpu")
        return self.q_network(x).to("cpu")

    def get_target_values(self, x):
        """
        The function `get_target_values` converts input `x` to a torch tensor of type float32 on CPU and
        returns the output of the target network also on CPU.
        
        :param x: input x
        :return: return the target values
        """
        x = torch.tensor(x).to(torch.float32).to("cpu")
        return self.target_network(x).to("cpu")

    def synchronize_networks(self):
        """
        The `synchronize_networks` function copies the state of the Q network to the target network in
        Python.
        """
        
        self.target_network.load_state_dict(self.q_network.state_dict())

    def get_action(self, x, greedy=True):
        """
        This function takes input data, calculates Q values, and returns the action with the highest Q
        value.
        
        :param x: input x
        :return: The `get_action` function returns the action with the highest Q-value based on the input
        state `x`. If `greedy` is set to True, it returns the action with the highest Q-value.
        """
        x = torch.tensor(x).to(torch.float32).to("cpu")
        q_vals = self.get_q_values(x)
        action =None
        if len(q_vals.shape) > 1:
            action = q_vals.argmax(dim=1)
        else:
            action = q_vals.argmax()
        return action