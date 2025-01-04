import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from torch.utils.data import DataLoader, TensorDataset


class ProbabilisticNeuralNetwork(nn.Module):
    """_summary_
    A probabilistic neural network that outputs both the mean and log standard deviation of a distribution.
    
    Args:
        input_dim (int): The input dimension for the network.
        hidden_dim (int): The number of hidden units in each layer.
        output_dim (int): The output dimension of the network.
        hidden_layers (int): The number of hidden layers.
        log_std_min (float): The minimum value for the log standard deviation.
        log_std_max (float): The maximum value for the log standard deviation.
        init_w (float): The weight initialization factor.
    
    Methods:
        init_weights(init_w):
            Initializes the weights of the network using kaiming uniform initialization.
        
        forward(x):
            Performs a forward pass of the network, returning the mean and log standard deviation of the output.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, hidden_layers, log_std_min=-20, log_std_max=1, init_w=9e-4):
        """_summary_
        Initializes the probabilistic neural network with specified dimensions and initialization.
        
        Args:
            input_dim (int): The input dimension for the network.
            hidden_dim (int): The number of hidden units in each layer.
            output_dim (int): The output dimension of the network.
            hidden_layers (int): The number of hidden layers.
            log_std_min (float): The minimum value for the log standard deviation.
            log_std_max (float): The maximum value for the log standard deviation.
            init_w (float): The weight initialization factor.
        """
        super(ProbabilisticNeuralNetwork, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.LeakyReLU())
        
        # Add hidden layers
        for _ in range(1, hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.LeakyReLU())
            
        # Output layer for both mean and log_std (output_dim * 2)
        layers.append(nn.Linear(hidden_dim, output_dim * 2))
        self.pnn = nn.Sequential(*layers)
        self.init_weights(init_w)
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def init_weights(self, init_w):
        """_summary_
        Initializes the weights of the network using kaiming uniform initialization.
        
        Args:
            init_w (float): The weight initialization factor.
        """
        for layer in self.pnn:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')

    def forward(self, x):
        """_summary_
        Performs a forward pass of the network, returning the mean and log standard deviation of the output.
        
        Args:
            x (Tensor): The input tensor.
        
        Returns:
            tuple: The mean and log standard deviation tensors.
        """
        print("Forward pass, " ,x)
        x = self.pnn(x)
        mean, log_std = torch.chunk(x, 2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        print("log_std ",log_std)
        #log_std = self.log_std_max - F.softplus(self.log_std_max - log_std)
        #log_std = self.log_std_min + F.softplus(log_std - self.log_std_min)
        return mean, log_std


class Ensemble(nn.Module):
    """_summary_
    An ensemble of probabilistic neural networks (PNNs) used to predict next state and reward in reinforcement learning.
    
    Args:
        env (gym.Env): The environment for which the ensemble is being used.
        learning_rate (float): The learning rate for the optimizer.
        num_steps (int): The number of total steps for training.
        hidden_dim (int): The number of hidden units in each PNN layer.
        num_ensembles (int): The number of ensemble members (PNNs).
        hidden_layers (int): The number of hidden layers in each PNN.
        prior_std (float): The standard deviation for the prior loss.
    
    Methods:
        forward(state, action):
            Performs a forward pass using the ensemble, returning the mean and log_std for the predicted next state and reward.
        
        sample_predictions(state, action):
            Samples predictions from each ensemble member and returns the mean next state and reward.
        
        loss(predicted_mus, predicted_log_vars, target_states):
            Computes the loss between predicted states and target states using MSE with inverse variance weighting.
        
        prior_loss(model):
            Computes the prior loss based on a Gaussian prior.
        
        train_step(data, epochs=10):
            Performs a training step on the ensemble using the provided data and number of epochs.
        
        linear_scheduler(epoch):
            A learning rate scheduler to gradually reduce the learning rate.
    """

    def __init__(self, env, learning_rate, num_steps, hidden_dim=5, num_ensembles=2, hidden_layers=2, prior_std=1.0):
        """_summary_
        Initializes the ensemble of probabilistic neural networks for model-based reinforcement learning.
        
        Args:
            env (gym.Env): The environment for which the ensemble is being used.
            learning_rate (float): The learning rate for the optimizer.
            num_steps (int): The number of total steps for training.
            hidden_dim (int): The number of hidden units in each PNN layer.
            num_ensembles (int): The number of ensemble members (PNNs).
            hidden_layers (int): The number of hidden layers in each PNN.
            prior_std (float): The standard deviation for the prior loss.
        """
        super(Ensemble, self).__init__()
        self.low = env.single_action_space.low
        self.high = env.single_action_space.high
        self.input_dim = np.prod(env.observation_space.shape) + np.prod(env.action_space.shape)
        self.output_dim = np.prod(env.observation_space.shape) + 1  # Next state + reward

        # Create an ensemble of PNNs
        self.ensembles = nn.ModuleList([ProbabilisticNeuralNetwork(self.input_dim, hidden_dim, self.output_dim, hidden_layers) for _ in range(num_ensembles)])
        self.num_ensembles = num_ensembles

        self.optimizers = [torch.optim.Adam(model.parameters(), lr=learning_rate) for model in self.ensembles]
        self.total_timesteps = num_steps
        self.schedulers = [torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=self.linear_scheduler) for optimizer in self.optimizers]

        # Variables for elite model selection
        self.best_loss = [float('inf')] * num_ensembles
        self.elite_models = list(range(num_ensembles))  # Start with all models as elite
        self.num_elites = max(1, num_ensembles // 2)  # Number of elite models to select
        self.device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.prior_std = prior_std
        
    def forward(self, state, action):
        """_summary_
        Performs a forward pass using the ensemble, returning the mean and log_std for the predicted next state and reward.
        
        Args:
            state (Tensor): The current state tensor.
            action (Tensor): The action tensor.
        
        Returns:
            tuple: A tuple containing the predicted means and log standard deviations.
        """
        if not torch.is_tensor(state):
            state = torch.tensor(state)
        
        if state.ndim > 2 and state.size(1) == 1:
            state = state.squeeze(1)
        if not torch.is_tensor(action):
            state = torch.tensor(action)
        
        if action.ndim > 2 and action.size(1) == 1:
            action = action.squeeze(1)
        
        obs_mean, obs_std = torch.mean(state, axis=0), torch.std(state, axis=0)
        action_mean, action_std = torch.mean(action, axis=0), torch.std(action, axis=0)
        epsilon = 1e-8

        #state = (state - obs_mean) / (obs_std + epsilon)
        #action = (action - action_mean) / (action_std + epsilon)
        print("input state, ",state)
        print("input action, ",action)
        inputs = torch.cat([state.to(torch.float32), action], dim=-1)
        means, log_stds = [], []
        for i in self.elite_models:
            print("input shape, ",inputs.squeeze(1))
            mean, log_std = self.ensembles[i](inputs.squeeze(1))
            means.append(mean)
            log_stds.append(log_std)

        means = torch.stack(means)
        log_stds = torch.stack(log_stds)
        return means, log_stds

    def sample_predictions(self, state, action):
        """_summary_
        Samples predictions from each ensemble member and returns the mean next state and reward.
        
        Args:
            state (Tensor): The current state tensor.
            action (Tensor): The action tensor.
        
        Returns:
            tuple: The predicted next state and reward.
        """
    
        print("state shape: ", state)
        print("action shape: ", action)  
        state_mean = state.mean(dim=(0, 1), keepdim=True)
        state_std = state.std(dim=(0, 1), keepdim=True) + 1e-8  # Avoid division by zero
        state_normalized = (state - state_mean) / state_std
        means, log_stds = self.forward(state_normalized, action)
   
        # Sample from each ensemble member
        samples = []
        for i in range(len(self.elite_models)):
            mean = means[i]
            
            #values turn NaN if I dont clip. Check the source of the issue
            std = torch.exp(torch.clamp(log_stds[i], min=-10, max=1))
            print("std: ",std)
            samples.append(torch.normal(mean, std))
        
        samples = torch.stack(samples)
        # Use the mean of the sampled predictions as the final prediction
        mean_prediction = samples.mean(dim=0)
        next_state = mean_prediction[..., :-1]
        reward = mean_prediction[..., -1:]
        return next_state, reward

    def loss(self, predicted_mus, predicted_log_vars, target_states):
        """_summary_
        Computes the loss between predicted states and target states using MSE with inverse variance weighting.
        
        Args:
            predicted_mus (Tensor): The predicted mean states.
            predicted_log_vars (Tensor): The predicted log variance for states.
            target_states (Tensor): The target states.
        
        Returns:
            Tensor: The computed state loss.
        """
        predicted_states = predicted_mus[:, :-1]
        target_states = target_states.squeeze(1)
        predicted_log_vars_states = predicted_log_vars[:, :-1]
        inv_var = torch.exp(-predicted_log_vars_states)

        mse_loss = nn.MSELoss(reduction='none')
        state_loss = 0.5 * (mse_loss(predicted_states, target_states) * inv_var + predicted_log_vars_states)

        return state_loss.mean()

    def prior_loss(self, model):
        """_summary_
        Computes the prior loss based on a Gaussian prior.
        
        Args:
            model (nn.Module): The model for which the prior loss is being computed.
        
        Returns:
            Tensor: The computed prior loss.
        """
        prior_loss = 0.0
        for param in model.parameters():
            prior_loss += torch.sum(param**2) / (2 * self.prior_std**2)
        return prior_loss

    def train_step(self, data, epochs=10):
        """_summary_
        Performs a training step on the ensemble using the provided data and number of epochs.
        
        Args:
            data (object): The training data object containing observations, actions, and target states.
            epochs (int): The number of epochs for training.
        
        Returns:
            tuple: The best loss and number of epochs completed.
        """
        obs = torch.tensor(data.observations, dtype=torch.float32).to(self.device)
        action = torch.tensor(data.actions, dtype=torch.float32).to(self.device)
        targets = torch.tensor(data.next_observations, dtype=torch.float32).to(self.device)

        dataset = TensorDataset(obs, action, targets)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

        for epoch in range(epochs):
            epoch_losses = []
            i = 0
            for model, optimizer, scheduler in zip(self.ensembles, self.optimizers, self.schedulers):
                model.train()  # Ensure the model is in training mode

                for batch in dataloader:
                    obs_batch, action_batch, targets_batch = batch
                    optimizer.zero_grad()

                    predicted_mus, predicted_log_vars = model(torch.cat((obs_batch.squeeze(1), action_batch), dim=-1))
                    likelihood_loss = self.loss(predicted_mus, predicted_log_vars, targets_batch)

                    # Combine the likelihood loss with the prior loss to get the MAP loss
                    prior_loss = self.prior_loss(model)
                    map_loss = likelihood_loss + prior_loss

                    map_loss.backward()
                    optimizer.step()

                    epoch_losses.append(map_loss.item())
                    scheduler.step()

                # Update the best loss for each model
                self.best_loss[i] = min(self.best_loss[i], np.mean(epoch_losses))
                i += 1
            elite_indices = np.argsort(self.best_loss)[:self.num_elites]
            self.elite_models = elite_indices

            # Additional logging
            avg_loss = np.mean(epoch_losses)
            print(f"Epoch {epoch+1}/{epochs}, Average MAP Loss: {avg_loss:.4f}")

        return self.best_loss, epochs

    def linear_scheduler(self, epoch):
        """_summary_
        A learning rate scheduler to gradually reduce the learning rate.
        
        Args:
            epoch (int): The current training epoch.
        
        Returns:
            float: The adjusted learning rate.
        """
        total_time_steps = self.total_timesteps
        return max(0.1, 1 - epoch / total_time_steps)
