import torch
import numpy as np
device = torch.device( "cpu")

def get_noisy_action(actor,obs,env,sigma):
    with torch.no_grad():
        #deterministic action
        action = actor(torch.Tensor(obs).to(device))
        # noise based on Ornstein-Uhlenback process
        noise = np.random.normal(scale = sigma, size = env.action_space.shape)
        # add noise to action
        noisy_action = action + noise
        #clip action if not in action bounds
        noisy_action = np.clip(noisy_action,env.action_space.low,env.action_space.high)
    return noisy_action

def get_noisy_action_target(actor,obs,env, sigma_d, noise_clip):
    with torch.no_grad():
        #deterministic action
        action = actor(torch.Tensor(obs).to(device))
        # noise based on Ornstein-Uhlenback process
        noise = np.random.normal(scale = sigma_d, size = env.action_space.shape)
        # add noise to action
        noisy_action = action + noise
        #clip action if not in action bounds
        noisy_action = np.clip(noisy_action,-noise_clip,noise_clip)
    return noisy_action
