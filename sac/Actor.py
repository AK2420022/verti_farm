import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
class Actor(nn.Module):
    def __init__(self, env, hidden_dim, log_std_min =-20, log_std_max = 2,init_w=3e-3):
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
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.l1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_layer = nn.Linear(hidden_dim,np.array(env.action_space.shape).prod())
        self.mean_layer.weight.data.uniform_(-init_w, init_w)
        self.mean_layer.bias.data.uniform_(-init_w, init_w)
        self.log_std_layer = nn.Linear(hidden_dim,np.array(env.action_space.shape).prod())
        self.log_std_layer.weight.data.uniform_(-init_w, init_w)
        self.log_std_layer.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        """
        The `forward` function takes a state as input, passes it through two ReLU layers and outputs the
        mean and log standard deviation of the distribution.
        
        :param state: Input state
        :return: The `forward` method returns the mean and the log standard deviation (clamped between
        `self.log_std_min` and `self.log_std_max`) of a neural network model.
        """
        a = F.relu(self.l1(torch.tensor(state).to("cpu")),inplace=True)
        a = F.relu(self.l2(a),inplace=True)
        mean = self.mean_layer(a)
        log_std = self.log_std_layer(a)
        log_std_layer= torch.clamp(log_std, self.log_std_min,self.log_std_max )
        return mean, log_std_layer
    def get_action(self,env,state,deterministic, explore,epsilon = 1e-6 ):
        """
        The `get_action` function in Python calculates an action and its log probability based on a given
        state and environment, with options for deterministic or stochastic behavior.
        
        :param env: Current environment
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
        mean,log_std = self.forward(torch.tensor(state).to("cpu"))
        
        std = log_std.exp()

        distribution = torch.distributions.Normal(0,1)
        action = None
        if deterministic : 
            action = distribution.mean
        else:
            action = distribution.rsample()
        
        log_prob = torch.distributions.Normal(mean, std).log_prob(mean + action * std)
        action = torch.tanh(mean + action * std)
        if len(log_prob.shape) > 1:
            log_prob = log_prob - torch.log((1-action.pow(2))+epsilon).sum(-1,keepdim=True)
        else:
            log_prob = log_prob - torch.log(1-action.pow(2)+epsilon).sum(-1,keepdim=True)
       
        return action, log_prob