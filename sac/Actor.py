import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
class Actor(nn.Module):
    def __init__(self, env, hidden_dim, log_std_min =-20, log_std_max = 2,init_w=3e-3):
        super(Actor, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.l1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_layer = nn.Linear(hidden_dim,np.array(env.action_space.shape).prod())
        self.mean_layer.weight.data.uniform_(-init_w, init_w)
        self.mean_layer.bias.data.uniform_(-init_w, init_w)
        self.log_std_layer = nn.Linear(hidden_dim,np.array(env.action_space.shape).prod())#nn.Parameter(torch.ones(1) * log_std_init, requires_grad=True)
        self.log_std_layer.weight.data.uniform_(-init_w, init_w)
        self.log_std_layer.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        a = F.relu(self.l1(torch.tensor(state).to("cpu")),inplace=True)
        a = F.relu(self.l2(a),inplace=True)
        mean = self.mean_layer(a)
        log_std = self.log_std_layer(a)
        #mean = torch.clamp(mean, -self.log_std_min,self.log_std_max )
        log_std_layer= torch.clamp(log_std, self.log_std_min,self.log_std_max )
        return mean, log_std
    def get_action(self,env,state,deterministic, explore,epsilon = 1e-6 ):
        #print("weights: ", self.mean_layer.weight)
        mean,log_std = self.forward(torch.tensor(state).to("cpu"))
        
        std = log_std.exp()

        distribution = torch.distributions.Normal(0,1)
        action = None
        if deterministic : 
            action = distribution.mean
        else:
            action = distribution.rsample()
        #action = torch.clamp(action,torch.tensor(env.action_space.low).to("cpu"),torch.tensor(env.action_space.high).to("cpu"))
        
        log_prob = torch.distributions.Normal(mean, std).log_prob(mean + action * std)
        action = torch.tanh(mean + action * std)
        if len(log_prob.shape) > 1:
            log_prob = log_prob - torch.log((1-action.pow(2))+epsilon).sum(-1,keepdim=True)
        else:
            log_prob = log_prob - torch.log(1-action.pow(2)+epsilon).sum(-1,keepdim=True)
       
        return action, log_prob