import torch
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_noisy_action(actor,obs,env,noisy_process, noise_scale, noisep):
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
    curr_noise += noisep.dt * (noisep.theta * (noisep.mu - curr_noise) + noisep.sigma * np.sqrt(noisep.dt) * np.random.normal(size=env.action_space.shape))
    return curr_noise

def evaluate_agent(envs, model, run_count, seed, greedy_actor=False):
    """
    Evaluate an agent on vectorized environment and return the returns and episode lengths of each run
    Args:
        envs: vectorized gym environment
        model: agent's policy model
        run_count: integer value specifying the number of runs to evaluate the agent
        seed: integer value representing the initial random seed
        greedy_actor: boolean flag that controls whether to use a greedy policy or not

    Returns:
        returns_over_runs: list of floats, representing the return of each run
        episode_len_over_runs: list of integers, representing the episode length of each run
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
