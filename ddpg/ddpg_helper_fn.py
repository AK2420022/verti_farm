import torch
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_noisy_action(actor,obs,env,noisy_process, noise_scale, noisep):
    """
    The function `get_noisy_action` generates a noisy action by adding noise based on an
    Ornstein-Uhlenbeck process to a deterministic action and then clipping the result within action
    bounds.
    
    :param actor: The `actor` model
    :param obs: the observation or state of the environment at a given time step. 
    :param env: The environment at a given time step
    :param noisy_process: the Ornstein-Uhlenbeck process. 
    :param noise_scale: The `noise_scale` parameter to scale the noise 
    :param noisep: The `noisep` parameter seems to be an object that contains the parameters for the
    Ornstein-Uhlenbeck process.
    :return: The function `get_noisy_action` returns the noisy action, 
    """
    with torch.no_grad():
        #deterministic action
        action = actor(torch.Tensor(obs).to(device))
        # noise based on Ornstein-Uhlenback process
        noise = noisep.theta * (noisep.mu - noisy_process) * noisep.dt + noisep.sigma * np.sqrt(noisep.dt) * np.random.normal(size = env.action_space.shape)
        # add noise to action
        noisy_action = action + noise_scale * noise
        #clip action if not in action bounds
        noisy_action = np.clip(noisy_action,env.action_space.low,env.action_space.high)
    return noisy_action

def update_noisy_process(curr_noise,env,noisep):
    """
    The function `update_noisy_process` updates the current noise level based on a stochastic process
    with given parameters.
    
    :param curr_noise: The `curr_noise` parameter represents the current noise level in a process.
    :param env: the environment in which the noisy process is being updated.
    :param noisep: The noise parameters
    :return: The function `update_noisy_process` returns the updated value of `curr_noise` after
    applying the noise process formula.
    """
    
    curr_noise += noisep.dt * (noisep.theta * (noisep.mu - curr_noise) + noisep.sigma * np.sqrt(noisep.dt) * np.random.normal(size=env.action_space.shape))
    return curr_noise

def evaluate_agent(envs, model, run_count, seed, greedy_actor=False):
    """
    This function evaluates an agent's performance in multiple environments using a given model for a
    specified number of runs.
    
    :param envs: The environment to evaluate
    :param model: The model to evaluate
    :param run_count: The `run_count` parameter in the `evaluate_agent` function represents the number
    of runs or episodes you want the agent to perform before returning the results. 
    :param seed: The `seed` parameter in the `evaluate_agent` function is used to set the seed value for
    the environment reset. This allows for reproducibility of the environment's initial state across
    different runs
    :param greedy_actor: The `greedy_actor` parameter in the `evaluate_agent` function is a boolean flag
    that determines whether the agent should act greedily based on the model's predictions. If
    `greedy_actor` is set to `True`, the agent will choose the action with the highest predicted value.
    If set, defaults to False (optional)
    :return: The function `evaluate_agent` returns two lists: `returns_over_runs` and
    `episode_len_over_runs`.
    """

    next_obs = torch.Tensor(envs.reset()[0])
    returns_over_runs = []
    episode_len_over_runs = []
    finish = False
    envs.reset(seed=list(range(seed, seed+envs.num_envs)))
    model.eval()
    while not finish:
        with torch.no_grad():
            actions = model(next_obs.cpu() )
        next_obs, rewards,_, _, infos = envs.step(actions.cpu().numpy().reshape(1,1))
        next_obs = torch.Tensor(next_obs).to("cuda" if torch.cuda.is_available() else "cpu")
        
        for info,values in infos.items():
            if "final_info" == info:
                returns_over_runs.append(values[0]["episode"]["r"])
                episode_len_over_runs.append(values[0]["episode"]["l"])
        if run_count==len(returns_over_runs):
            finish = True
            break
    model.train()
    return returns_over_runs, episode_len_over_runs
