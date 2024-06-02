import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from Actor import Actor
# This class represents a probabilistic neural network in Python using PyTorch.
class ProbabilisticNeuralNetwork(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim,hidden_layers,log_std_min=-10, log_std_max=2,init_w=3e-3):
        super(ProbabilisticNeuralNetwork, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LeakyReLU())
        # Add hidden layers
        for i in range(1, hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LeakyReLU())
        
        # Add output layer
        layers.append(nn.Linear(hidden_dim, output_dim*2) )
        self.pnn = nn.Sequential(*layers)
        #self.pnn.apply(self.weight_init)
        self.init_weights(init_w)
        self.log_std_min = -20 
        self.log_std_max = 2 
    def init_weights(self, init_w):
        for layer in self.pnn:
            if isinstance(layer, nn.Linear):
                nn.init.uniform_(layer.weight, -init_w, init_w)
                nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        """
        The `forward` function takes an input `x`, processes it through a neural network `pnn`, and returns
        the mean and log standard deviation after some transformations.
        
        :param x:Input x
        """
        x = self.pnn(x)
        mean, log_std = torch.chunk(x, 2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        log_std = self.log_std_max - F.softplus(self.log_std_max - log_std)
        log_std = self.log_std_max + F.softplus(log_std - self.log_std_max)
        return mean, log_std

class Ensemble(nn.Module):
    def __init__(self, env,learning_rate , hidden_dim=256, num_ensembles=4,hidden_layers=4):
        super(Ensemble, self).__init__()
        self.low = env.single_action_space.low
        self.high = env.single_action_space.high
        self.input_dim = np.array(env.observation_space.shape).prod() + np.array(env.action_space.shape).prod()
        self.output_dim = np.array(env.observation_space.shape).prod() + 1
        self.ensembles = nn.ModuleList([ProbabilisticNeuralNetwork(self.input_dim, hidden_dim, self.output_dim,hidden_layers) for _ in range(num_ensembles)])
        self.num_ensembles = num_ensembles
        self.optimizers = [torch.optim.Adam(model.parameters(), lr=learning_rate) for model in self.ensembles]
        self.schedulers = [torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=self.linear_scheduler) for optimizer in self.optimizers]
        self.best_loss = [1e10 for i in range(num_ensembles)]
        self.improvement_threshold = 0.01
        self.max_no_improvements = 5
        self.num_elites = 5
        self.elite_models = self.ensembles
        self.elite_optimizers =self.optimizers
    def forward(self, state, action):
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
        q = torch.cat([torch.tensor(state).float(), torch.tensor(action).float()], dim=-1)
        q = [ensemble(q) for ensemble in self.elite_models]
        means, log_stds = zip(*q)
        means = torch.stack(means)
        log_stds = torch.stack(log_stds)
        return means, log_stds

    def init_weights(self):
        """
        The `init_weights` function initializes the weights of linear layers in ensembles using a specified
        initialization method.
        
        """
        init_w = 0.001
        for layer in self.ensembles:
            for l in layer.pnn:
                if isinstance(l, nn.Linear):
                    #nn.init.xavier_uniform_(l.weight)
                    l.weight.data.uniform_(-init_w, init_w)
                    l.weight.data.uniform_(-init_w, init_w)

    def sample_predictions(self, state, action):
        """
        This function generates sample predictions using a random model from an ensemble and samples an
        action from the chosen model's distribution.
        
        :param state: Input current state
        :param action: Input actions 
        :return: The `sample_predictions` function returns the next state and reward.
        """
        sample_means = []
        sample_stds = []
        means, log_stds = self(state, action)
        # Extract means and log_stds from predictions
        for i in range(self.num_elites):
            mean = means[i]
            std = torch.exp(log_stds[i])
            sample_means.append(mean)
            sample_stds.append(std)

        # Stack means and stds
        sample_means = torch.stack(sample_means)
        sample_stds = torch.stack(sample_stds)
        
        # Choose a random model from the ensemble
        chosen_model = np.random.randint(0, self.num_elites)
        sample_mean = sample_means[chosen_model]
        sample_std_ = sample_stds[chosen_model][:,:,1:].sqrt() 
        sample_mean_= sample_mean[:,:,1:] 
        sample_mean_action =  sample_mean[:,:,:1] 
        #sample_std_action = sample_stds[chosen_model][:,:,:1].exp().sqrt() 
        #next_state= state + torch.distributions.Normal(sample_mean_,sample_std_).sample()
        next_state= torch.distributions.Normal(sample_mean_,sample_std_).sample()
        #print("next_state, ",next_state)

        reward = torch.distributions.Normal(sample_mean_action,1).sample()
        return next_state, reward#.squeeze(1)
    
    def loss(self, predicted_mus, predicted_log_vars, target_states):
        """
        This function calculates the loss using predicted means and log variances, selects elite models
        based on the loss values, and returns the mean loss.
        
        :param predicted_mus: The `predicted_mus` parameter seems to represent the predicted means from a
        model. 
        :param predicted_log_vars: The `predicted_log_vars` parameter in the `loss` function represents the
        predicted logarithm of the variance for each ensemble model.
        :param target_states: `target_states` is a tensor representing the target states for the model.
        :return: the mean of the losses calculated for the predicted values compared to the target states.
        """
        target_states = target_states#.unsqueeze(0).expand(self.num_ensembles, -1, -1)
        mse_loss = nn.MSELoss(reduction='none')
        inv_var = torch.exp(-predicted_log_vars)
        losses = mse_loss(predicted_mus, target_states) * inv_var + predicted_log_vars
        losses =losses.squeeze(1).mean(dim=0)

        return losses.mean()
    def linear_scheduler(self,epoch,total_time_steps = 20000):
        """
        Returns the linear decay value for the learning rate scheduler.

        :param epoch: The current epoch number.
        :return: The decay value based on the current epoch and total timesteps.

        This method computes a linear decay value that decreases from 1 to 0 over the total number of timesteps.
        """
        return 1 - epoch / total_time_steps
    def train(self, data,epochs=10):
        """
        This function to train the ensemble 
        
        :param data: The sampled data from environment that is used to train the model
        :param epochs: Number of epochs to train
        :param target_states: `target _states` is a tensor representing the target states for the model.
        :return: the mean of the losses calculated for the predicted values compared to the target states.
        """
        obs = torch.tensor(data.observations, dtype=torch.float32)
        action = torch.tensor(data.actions, dtype=torch.float32)
        targets = torch.tensor(data.next_observations, dtype=torch.float32)
        action = action[:,:,np.newaxis]
        losses = []
        model_no = 0
        #for epoch in range(epochs):
        for model, optimizer,scheduler in zip(self.elite_models, self.optimizers,self.schedulers):
            predicted_mus, predicted_log_vars = self.forward(obs, action)
            loss = self.loss(predicted_mus[model_no,:,:,1:], predicted_log_vars[model_no,:,:,1:], targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.detach())
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print("model_no,:", model_no)
            print("current_lr,: ",current_lr)
            #self.print_param_values(model)
            model_no +=1
            if model_no>=self.num_elites:
                break
        elite_indices = np.argsort(np.array(losses))[:self.num_elites]
        models = []
        optimizers = []
        for i, l in enumerate(self.ensembles):
            for idx in elite_indices:
                if idx == i : 
                    model = self.ensembles[i]
                    optimizer= self.optimizers[i]
                    models.append(model)
                    optimizers.append(optimizer)

        self.elite_models = nn.ModuleList(models)
        self.optimizers = optimizers
        return loss, epochs
    def print_param_values(self,model):
        """
        Prints the values of the parameters of the given model.

        :param model: The model whose parameters are to be printed.

        This method iterates over the named parameters of the model and prints the name and value of each parameter.
        """
        for name, param in model.named_parameters():
            print(f"Parameter {name} value: {param.data}")
        for name, param in model.named_parameters():
            if param.grad is not None:
                print(f"Gradient for {name}:")
                print(param.grad)
            else:
                print(f"No gradient for {name}")
