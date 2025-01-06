import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SquashBijector:
    """_summary_
    A class that provides a squashing bijector for transforming values using the tanh function and calculating log probabilities.
    
    Methods:
        forward(x):
            Applies the squashing transformation using tanh.
        
        inverse(y, high):
            Applies the inverse transformation.
        
        log_prob(log_prob, x):
            Adjusts the log probability based on the squashing transformation.
    """

    @staticmethod
    def forward(x):
        """_summary_
        Applies the squashing transformation using tanh.
        
        Args:
            x (torch.Tensor): Input tensor to apply tanh.
        
        Returns:
            torch.Tensor: Squashed output.
        """
        return torch.tanh(x)

    @staticmethod
    def inverse(y, high):
        """_summary_
        Applies the inverse transformation of the squashing function.
        
        Args:
            y (torch.Tensor): Output of the squashing function.
            high (float): The upper bound for scaling.
        
        Returns:
            torch.Tensor: The transformed input before squashing.
        """
        intermediate_result = 0.5 * torch.log((1 + y) / (1 - y))
        return intermediate_result

    @staticmethod
    def log_prob(log_prob, x):
        """_summary_
        Adjusts the log probability based on the squashing transformation.
        
        Args:
            log_prob (torch.Tensor): The initial log probability.
            x (torch.Tensor): The input tensor for which the log probability is adjusted.
        
        Returns:
            torch.Tensor: The adjusted log probability.
        """
        log_prob -= torch.sum(torch.log(1 - torch.tanh(x)**2 + 1e-6), dim=-1, keepdim=True)
        return log_prob


class Actor(nn.Module):
    """_summary_
    Actor Network for reinforcement learning, which outputs actions based on the given state.
    
    Args:
        env (gym.Env): The environment used for the agent.
        hidden_dim (int): The number of hidden units in each hidden layer.
        hidden_layers (int): The number of hidden layers.
        log_std_min (float, optional): The minimum log standard deviation. Defaults to -20.
        log_std_max (float, optional): The maximum log standard deviation. Defaults to 1.
    
    Methods:
        init_weights():
            Initializes the weights using orthogonal initialization.
        
        forward(state):
            Computes the action mean and log standard deviation for a given state.
        
        get_action(state, deterministic=False, epsilon=1e-6):
            Returns the action for a given state, optionally using a deterministic policy.
    """

    def __init__(self, env, hidden_dim, hidden_layers, log_std_min=-20, log_std_max=1):
        """_summary_
        Initializes the Actor network.
        
        Args:
            env (gym.Env): The environment used for the agent.
            hidden_dim (int): The number of hidden units in each hidden layer.
            hidden_layers (int): The number of hidden layers.
            log_std_min (float, optional): The minimum log standard deviation. Defaults to -20.
            log_std_max (float, optional): The maximum log standard deviation. Defaults to 1.
        """
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
        """_summary_
        Initializes the weights using orthogonal initialization for the linear layers.
        """
        for layer in self.actor:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)

    def forward(self, state):
        """_summary_
        Computes the action mean and log standard deviation for a given state.
        
        Args:
            state (torch.Tensor or np.ndarray): The input state tensor.
        
        Returns:
            mean (torch.Tensor): The mean of the action distribution.
            log_std (torch.Tensor): The log standard deviation of the action distribution.
        """
        if not torch.is_tensor(state):
            state = torch.tensor(state)
        
        if state.ndim > 2 and state.size(1) == 1:
            state = state.squeeze(1)
        
        state = torch.tensor(state).to(torch.float32).to(device)
        a = self.actor(state)
        mean, log_std = torch.chunk(a, 2, dim=-1)
        return mean, log_std

    def get_action(self, state, deterministic=False, epsilon=1e-6):
        """_summary_
        Returns the action for a given state, optionally using a deterministic policy.
        
        Args:
            state (torch.Tensor or np.ndarray): The input state tensor.
            deterministic (bool, optional): Whether to use a deterministic policy. Defaults to False.
            epsilon (float, optional): A small value to avoid division by zero in log probability. Defaults to 1e-6.
        
        Returns:
            action (torch.Tensor): The action chosen for the given state.
            log_prob (torch.Tensor): The log probability of the chosen action.
        """
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
