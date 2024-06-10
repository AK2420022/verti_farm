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
    def __init__(self, env, hidden_dim,hidden_layers, log_std_min=-20, log_std_max=2):
        """
        This function initializes the parameters for an actor neural network used in reinforcement learning,
        setting up the layers for mean and log standard deviation calculations.
        
        :param env: The environment to 
        :param hidden_dim: the size of the hidden layers in the neural network architecture
        :param log_std_min:the minimum value for the logarithm of the standard deviation. This parameter is
        initialized with a default value of -20
        :param log_std_max: the maximum value for the logarithm of the standard deviation in a probabilistic policy.  defaults to
        2 (optional)
        :param init_w: the range for initializing the weights of the mean and log standard deviation layers.
        """
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
       
    def init_weights(self):
        """
        The `init_weights` function initializes the weights of linear layers in ensembles using a specified
        initialization method.
        
        """
        init_w = 0.01
        for layer in self.actor:
            if isinstance(layer, nn.Linear):
                layer.weight.data.uniform_(-init_w , init_w)
                layer.weight.data.uniform_(-init_w, init_w)
    def forward(self, state):
        """
        The `forward` function takes a state and an action as input, concatenates them, passes them through
        a list of ensemble models, and returns the means and log standard deviations of the output
        distributions.
        
        :param state: Input state
        :return: The `forward` method returns two tensors: `means` and `log_stds`. `means` is a tensor
        containing the means calculated from the ensemble models for the given state and action, and
        `log_stds` is a tensor containing the log standard deviations calculated from the ensemble models
        for the given state and action.
        """
        a = self.actor(torch.tensor(state).to(torch.float32))
        mean, log_std = torch.chunk(a, 2, dim=-1)
        #log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        log_std = self.log_std_max - F.softplus(self.log_std_max - log_std)
        log_std = self.log_std_min + F.softplus(log_std - self.log_std_min)
        return mean, log_std

    def get_action(self, state, deterministic=False, epsilon=1e-6):
        """
        The `get_action` function in Python calculates an action and its log probability based on a given
        state and environment, with options for deterministic or stochastic behavior.
        
        :param state: The current state
        :param deterministic: The `deterministic` parameter in the `get_action` method is a boolean flag
        that determines whether the action selection should be deterministic or stochastic
        :param explore: The `explore` parameter in the `get_action` method is used to determine whether the
        agent should explore or exploit in the context of reinforcement learning. When `explore` is set to
        `True`, the agent will choose actions that may not be optimal but help in exploring the environment
        to learn
        :param epsilon: The `epsilon` parameter in the `get_action` function is a small value used to
        prevent division by zero when calculating the logarithm of very small numbers. It is added to the
        denominator to ensure numerical stability in the computation. In this case, `epsilon` is set to 1e-6
        :return: The function `get_action` returns the action and the log probability of that action based
        on the input parameters and the calculations performed within the function.
        """
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
        #print("action, ", action)
        # Calculate log probability
        # Apply inverse transformation of squash bijector to calculate log_prob
        log_prob = distribution.log_prob(action_).sum(-1, keepdim=True)
        log_prob =  SquashBijector.log_prob(log_prob, action_)
        return action, log_prob