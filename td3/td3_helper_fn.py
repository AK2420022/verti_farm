import torch
import numpy as np
device = torch.device( "cpu")

def get_noisy_action(actor,obs,env,sigma):
    """
    The function `get_noisy_action` adds noise to a deterministic action using an Ornstein-Uhlenbeck
    process and clips the noisy action within the action bounds.
    
    :param actor: The `actor` model
    :param env: The current environment
    :param sigma: The `sigma` parameter in the `get_noisy_action` function represents the standard
    deviation of the noise added to the action. 
    :return: The noise added action
    """
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
    """
    The function `get_noisy_action` adds noise to a deterministic action using an Ornstein-Uhlenbeck
    process and clips the noisy action within the action bounds.
    
    :param actor: The `actor` model
    :param env: The current environment
    :param sigma_d: The `sigma` parameter in the `get_noisy_action` function represents the standard
    deviation of the noise added to the action. 
    :param noise_clip: The bounds for the noises
    :return: The noise added action
    """
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
