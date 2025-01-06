import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class Critic2(nn.Module):
    def __init__(self, env,hidden_dim):
        """
        This Python function defines a neural network model for a critic in reinforcement learning, with
        three linear layers.
        
        :param env: Environment to initialize
        :param hidden_dim: It is used to define the size of the hidden layers in the network architecture
        """
        super(Critic2, self).__init__()
        self.l1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + np.array(env.action_space.shape).prod() , hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, np.array(env.action_space.shape).prod())

    def forward(self, state, action):
        """
        This Python function takes a state and an action as input, outputs the q values
        
        :param state: The input state
        :param action: Input action
        :return: The predicated q values 
        """
        q =  torch.cat([torch.tensor(state).to(torch.float32), torch.tensor(action).to(torch.float32)], 1)
        a = F.relu(self.l1(q),inplace=True)
        a = F.relu(self.l2(a),inplace=True)
        return self.l3(a)
    
    def q1_forward(self, state, action):
        """
        This Python function takes a state and an action as input, outputs the q values 
        based only on one critic network
        
        :param state: The input state
        :param action: Input action
        :return: The predicated q values 
        """
        q =   torch.cat([torch.tensor(state).to(torch.float32), torch.tensor(action).to(torch.float32)], 1)
        q_values = self.critics[0](q)
        return q_values

class Critic1(nn.Module):
    def __init__(self, env,hidden_dim):
        super(Critic1, self).__init__()
        """
        This Python function defines a neural network model for a critic in reinforcement learning, with
        three linear layers.
        
        :param env: Environment to initialize
        :param hidden_dim: It is used to define the size of the hidden layers in the network architecture
        """
        self.l1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + np.array(env.action_space.shape).prod() , hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, np.array(env.action_space.shape).prod())

    def forward(self, state, action):
        """
        This Python function takes a state and an action as input, outputs the q values
        
        :param state: The input state
        :param action: Input action
        :return: The predicated q values 
        """
        q =  torch.cat([torch.tensor(state).to(torch.float32), torch.tensor(action).to(torch.float32)], 1)
        a = F.relu(self.l1(q),inplace=True)
        a = F.relu(self.l2(a),inplace=True)
        return self.l3(a)
    
    def q1_forward(self, state, action):
        """
        This Python function takes a state and an action as input, outputs the q values 
        based only on one critic network
        
        :param state: The input state
        :param action: Input action
        :return: The predicated q values 
        """
        q =   torch.cat([torch.tensor(state).to(torch.float32), torch.tensor(action).to(torch.float32)], 1)
        q_values = self.critics[0](q)
        return q_values