import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class Critic(nn.Module):
    """_summary_
    Critic neural network module used in reinforcement learning for estimating Q-values.
    
    Args:
        env (gym.Env): The environment for which the critic is being used.
        hidden_dim (int): The number of hidden units in each layer.
        hidden_layers (int): The number of hidden layers in the network.
        init_w (float): The weight initialization factor for the layers.
    
    Methods:
        init_weights(init_w1, init_w2):
            Initializes the weights of the Q-networks with kaiming normal initialization.
            
        forward(state, action):
            Takes a state and an action as input and outputs the Q-values from both critics.
            
        q1_forward(state, action):
            Takes a state and an action as input and outputs the Q-values from the q1 critic.
    """

    def __init__(self, env, hidden_dim, hidden_layers, init_w=0.01):
        """_summary_
        Initializes the Critic network with two Q-networks (q1 and q2).
        
        Args:
            env (gym.Env): The environment for which the critic is being used.
            hidden_dim (int): The number of hidden units in each layer.
            hidden_layers (int): The number of hidden layers in the network.
            init_w (float): The weight initialization factor for the layers.
        """
        super(Critic, self).__init__()

        # Define the input size
        input_size = np.array(env.single_observation_space.shape).prod() + np.array(env.action_space.shape).prod()

        # Create layers for q1
        layers_q1 = []
        layers_q1.append(nn.Linear(input_size, hidden_dim))
        layers_q1.append(nn.SiLU())
        for _ in range(1, hidden_layers):
            layers_q1.append(nn.Linear(hidden_dim, hidden_dim))
            layers_q1.append(nn.SiLU())
        layers_q1.append(nn.Linear(hidden_dim, 1))
        self.q1 = nn.Sequential(*layers_q1)

        # Create layers for q2
        layers_q2 = []
        layers_q2.append(nn.Linear(input_size, hidden_dim))
        layers_q2.append(nn.SiLU())
        for _ in range(1, hidden_layers):
            layers_q2.append(nn.Linear(hidden_dim, hidden_dim))
            layers_q2.append(nn.SiLU())
        layers_q2.append(nn.Linear(hidden_dim, 1))
        self.q2 = nn.Sequential(*layers_q2)

        # Initialize weights
        self.init_weights(init_w, init_w)

    def init_weights(self, init_w1, init_w2):
        """_summary_
        Initializes the weights of the Q-networks with kaiming normal initialization.
        
        Args:
            init_w1 (float): Weight initialization factor for q1 network.
            init_w2 (float): Weight initialization factor for q2 network.
        """
        for layer in self.q1:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
        for layer in self.q2:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, state, action):
        """_summary_
        Takes a state and an action as input and outputs the Q-values from both critics.
        
        Args:
            state (Tensor or np.array): The input state.
            action (Tensor or np.array): The input action.
        
        Returns:
            tuple: A tuple containing the predicted Q-values from both q1 and q2.
        """
        # Ensure tensors are float32
        state = torch.tensor(state, dtype=torch.float32).to(device)
        action = torch.tensor(action, dtype=torch.float32).to(device)

        if state.size(1) == 1:
            state = state.squeeze(1)
        if action.size(1) == 1:
            action = action.squeeze(1)
        
        q_input = torch.cat([state, action], dim=-1)
        return self.q1(q_input), self.q2(q_input)

    def q1_forward(self, state, action):
        """_summary_
        Takes a state and an action as input and outputs the Q-values from the q1 critic.
        
        Args:
            state (Tensor or np.array): The input state.
            action (Tensor or np.array): The input action.
        
        Returns:
            Tensor: The predicted Q-values from the q1 critic.
        """
        # Ensure tensors are float32
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32).to(device)
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32).to(device)

        q_input = torch.cat([state, action], dim=-1)
        return self.q1(q_input)
