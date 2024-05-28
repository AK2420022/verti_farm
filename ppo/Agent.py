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
        """
        The function initializes a neural network for Q-learning with a target network that is a deep copy
        of the Q-network.
        
        :param env:  The `env` parameter
        """
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
        """
        The function `get_value` takes an input `x` and returns the output of the `critic` function applied
        to `x` after converting it to a tensor.
        
        :param x: input x
        :return: The output of the `critic` function
        """
        return self.critic(torch.tensor(x))

    def get_action(self, x, greedy=False):
        """
        The function `get_action` returns a sampled action from a categorical distribution based on the
        logits generated by a neural network actor.
        
        :param x: Input x 
        :param greedy: The `greedy` parameter in the `get_action` method is a boolean flag that determines
        whether to select the action based on a greedy strategy or not. When `greedy` is set to `False`, the
        action is sampled from the distribution of logits generated by the actor network. However,, defaults
        to False (optional)
        :return: The `get_action` method returns an action based on the input `x`. 
        """
        logits = self.actor(torch.tensor(x))
        distribution = Categorical.Categorical(logits=logits)
        action = distribution.sample() if not greedy else distribution.mode
        return action

    def get_action_and_value(self, x, action=None):
        """
        The function `get_action_and_value` takes input `x` and an optional `action`, calculates the logits
        using a neural network, samples an action if not provided, and returns the action, log probability,
        entropy, and critic value.
        
        :param x: Input x
        :param action: The actions 
        :return: The function `get_action_and_value` returns four values:
        1. `action`: The sampled action from the probability distribution.
        2. `log_prob`: The log probability of the sampled action.
        3. `entropy`: The entropy of the probability distribution.
        4. `self.critic(x)`: The value estimated by the critic network for the input `x`.
        """
        logits = self.actor(torch.tensor(x))
        distr =Categorical.Categorical(logits=logits)
        if action is None:
            action = distr.sample()
        return action, distr.log_prob(action), distr.entropy(), self.critic(x)