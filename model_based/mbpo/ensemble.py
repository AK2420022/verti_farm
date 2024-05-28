import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from Actor import Actor
class ProbabilisticNeuralNetwork(nn.Module):
    def weight_init(self,m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
    def __init__(self, input_dim, hidden_dim, output_dim,hidden_layers,log_std_min=-10, log_std_max=2,init_w=3e-4):
        super(ProbabilisticNeuralNetwork, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.SiLU())
        # Add hidden layers
        for i in range(1, hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        
        # Add output layer
        layers.append(nn.Linear(hidden_dim, output_dim*2) )
        self.pnn = nn.Sequential(*layers)
        #self.pnn.apply(self.weight_init)

        self.log_std_min = -10 #nn.Parameter((-torch.ones((1, output_dim)).float() * 10), requires_grad=False)
        self.log_std_max = 2 #nn.Parameter((torch.ones((1, output_dim)).float() / 2), requires_grad=False)

    def forward(self, x):
        x = self.pnn(x)
        mean, log_std = torch.chunk(x, 2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        log_std = self.log_std_max - F.softplus(self.log_std_max - log_std)
        log_std = self.log_std_max + F.softplus(log_std - self.log_std_max)
        return mean, log_std
        #logvar = self.log_std_max - F.softplus(self.log_std_max - logvar)
        #logvar = self.log_std_min + F.softplus(logvar - self.log_std_min)
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
        self.best_loss = [1e10 for i in range(num_ensembles)]
        self.improvement_threshold = 0.01
        self.max_no_improvements = 5
        self.num_elites = 5
        self.elite_models = self.ensembles
        self.elite_optimizers =self.optimizers
    def forward(self, state, action):
        q = torch.cat([torch.tensor(state).float(), torch.tensor(action).float()], dim=-1)
        q = [ensemble(q) for ensemble in self.elite_models]
        means, log_stds = zip(*q)
        means = torch.stack(means)
        log_stds = torch.stack(log_stds)
        return means, log_stds

    def init_weights(self, init_method):
        init_w = 0.001
        for layer in self.ensembles:
            for l in layer.pnn:
                if isinstance(l, nn.Linear):
                    l.weight.data.uniform_(-init_w, init_w)
                    l.weight.data.uniform_(-init_w, init_w)

    def sample_predictions(self, actor , state, action):
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
        sample_std_ = sample_stds[chosen_model][:,:,1:].exp().sqrt() 
        sample_mean_= sample_mean[:,:,1:] 
        sample_mean_action =  sample_mean[:,:,:1] 
        sample_std_action = sample_stds[chosen_model][:,:,:1].exp().sqrt() 
        # Sample an action from the chosen model's distribution
        #print(state.shape)
        #print(sample_mean_.shape)
        next_state= state + torch.distributions.Normal(sample_mean_,sample_std_).sample()
        reward = torch.distributions.Normal(sample_mean_action,1).sample()
        #action, log_prob = actor.get_action(next_state)
        # Extract the desired action and reshape
        #action = torch.tensor(self.low + (self.high - self.low)) * (action + 1) / 2
        return next_state, reward#.squeeze(1)
    
    def loss(self, predicted_mus, predicted_log_vars, target_states):
        target_states = target_states.unsqueeze(0).expand(self.num_ensembles, -1, -1)
        mse_loss = nn.MSELoss(reduction='none')
        inv_var = torch.exp(-predicted_log_vars)
        losses = mse_loss(predicted_mus, target_states) * inv_var + predicted_log_vars
        
        losses =losses.mean(dim=1).sum(dim=1)
        
        elite_indices = np.argsort(losses.detach().numpy())[:self.num_elites]
        models = []
        optimizers = []
        for i, l in enumerate(self.ensembles):
            for idx in elite_indices:
                #print("i, ",i)
                #print("idx, ",idx)
                if idx == i : 
                    model = self.ensembles[i]
                    optimizer= self.optimizers[i]
                    models.append(model)
                    optimizers.append(optimizer)

        self.elite_models = nn.ModuleList(models)
        self.optimizers = optimizers
        return losses.mean()

    def train(self, data,epochs=10, batch_size=256):
        obs = torch.tensor(data.observations.squeeze(1), dtype=torch.float32)
        action = torch.tensor(data.actions, dtype=torch.float32)
        targets = torch.tensor(data.next_observations.squeeze(1), dtype=torch.float32)
        
        model_no = 0
        #for epoch in range(epochs):
        for model, optimizer in zip(self.elite_models, self.optimizers):
            optimizer.zero_grad()
            predicted_mus, predicted_log_vars = self.forward(obs, action)
            loss = self.loss(predicted_mus[model_no,:,1:], predicted_log_vars[model_no,:,1:], targets)
            loss.backward()
            optimizer.step()

            model_no +=1
            if model_no>=self.num_elites:
                break
        return loss, epochs
 