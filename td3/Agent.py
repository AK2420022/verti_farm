import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
device = torch.device("cpu")

# This Python class defines an actor model for reinforcement learning with three linear layers.
class Actor(nn.Module):
    def __init__(self, env):
        """
        This Python function initializes an Actor object with specific attributes related to the
        environment.
        
        :param env: The environment to initialize
        """
        super(Actor, self).__init__()
        self.action_dim = env.action_space.high
        self.l1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

    
    def forward(self, state):
        """
        The function takes a state as input, and predicts the action
        
        :param state: The state to evaluate action for
        :return: THe predicted action
        """
        a = F.relu(self.l1(torch.tensor(state).to(torch.float32).to("cpu")))
        a = F.relu(self.l2(a))
        return torch.tensor(self.action_dim).to("cpu") * torch.tanh(self.l3(a))
    

#Q1_theta1
# This class defines a Critic neural network with multiple critic modules for reinforcement learning
# tasks.
class Critic(nn.Module):
    def __init__(self, env, num_critics):
        """
        The function initializes a Critic class with multiple neural networks for evaluating the value of
        states in a reinforcement learning environment.
        
        :param env: The environment to initialize
        :param num_critics: The number of critics
        """
        super(Critic, self).__init__()
        self.l1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 400)
        self.l2 = nn.Linear(400 + 1, 300)
        self.l3 = nn.Linear(300, 1)
        
        self.critics = nn.ModuleList([
            nn.Sequential(
                nn.Linear(np.array(env.single_observation_space.shape).prod() + 1, 400),
                nn.ReLU(),
                nn.Linear(400, 300),
                nn.ReLU(),
                nn.Linear(300, 1)
            )
            for _ in range(num_critics)
        ])

    def forward(self, state, action):
        """
        The `forward` function takes a state and an action as input, concatenates them, passes them through
        multiple critic networks, and returns the Q-values predicted by each critic.
        
        :param state: Input states
        :param action: input actions
        :return: The predicted Q-values
        """
        q =  torch.cat([torch.tensor(state).clone().detach().requires_grad_(True).to(torch.float32), action.to(torch.float32)], 1)
        q_values = [critic(q) for critic in self.critics]
        return q_values
    
    def q1_forward(self, state, action):
        """
        The function `q1_forward` takes a state and an action as input, concatenates them, passes them
        through a neural network, and returns the output Q values based on the first crtiic only.
        
        :param state: Input states
        :param action: input actions
        :return: The predicted Q-values
        """
        q =  torch.cat([torch.tensor(state.clone().detach().requires_grad_(True)).to(torch.float32), action.to(torch.float32)], 1)
        q_values = self.critics[0](q)
        return q_values