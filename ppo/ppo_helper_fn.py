
import numpy as np
import torch
def compute_advantage_estimates(rewards, values, dones, next_value, next_done,hypp):
    '''
    Computes the estimated advantages from given rewards and estimated values
    :param rewards: accumulated rewards during the rollout (shape [num_steps, num_envs])
    :param values: estimated values of traversed states (shape [nums_steps, num_envs])
    :param next_value: bootstrapped value of the next state after rollout (shape [1, num_envs])
    :param next_done: whether environment is finished in next state after rollout (shape [1, num_envs])
    :return: the estimated advantages of policy (shape [num_steps, num_envs])
    '''
    with torch.no_grad():
        returns = torch.zeros_like(rewards)
        for t in reversed(range(hypp.num_steps)):
            if t == hypp.num_steps - 1:
                nextnonterminal = 1.0 - next_done
                returns[t] = rewards[t] + hypp.gamma * next_value * nextnonterminal
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                returns[t] = rewards[t] + hypp.gamma * values[t+1] * nextnonterminal 
        advantages = returns - values

    return advantages

def compute_gae(rewards, values, dones, next_value, next_done, gae_lambda,hypp):
    '''
    Computes the generalized advantage estimates(GAE) for every timestep from given rewards and estimated values
    :param rewards: accumulated rewards during the rollout (shape [num_steps, num_envs])
    :param values: estimated values of traversed states (shape [num_steps, num_envs])
    :param next_value: bootstrapped value of the next state after rollout (shape [1, num_envs])
    :param next_done: whether environment is finished in next state after rollout (shape [1, num_envs])
    :param gae_lambda: scalar coefficient for gae computation
    :return: generalized advantage estimates of trajectory (shape [num_steps, num_envs])
    '''
    # TODO: Part b)
    
    
    with torch.no_grad():
        last_a = 0
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        for t in reversed(range(hypp.num_steps)):
            if t == hypp.num_steps - 1:
                nextnonterminal = 1.0 - next_done
                returns[t] = rewards[t] + hypp.gamma * next_value * nextnonterminal - values[t]
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                returns[t] = rewards[t] + hypp.gamma * values[t+1] * nextnonterminal - values[t]
            advantages[t] = returns[t] + gae_lambda * hypp.gamma* nextnonterminal * last_a
            last_a = advantages[t]
    
    return advantages

def normalize_advantages(advantages):
    '''
    Takes tensor of advantages and normalizes them to zero mean and unit variance
    :param advantages: tensor of advantages to normalize (shape [n])
    :return: tensor of normalized advantages (shape [n])
    '''
    # TODO: Part b)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) 
    return advantages

def compute_policy_objective(advantages, logprob_old, logprob,hypp):
    '''
    Computes the policy objective that is being optimized in the gradient step.
    :param advantages: tensor of advantages (shape [n])
    :param logprob_old: tensor of log-probabilites of actions of sampled policy (shape [n])
    :param logprob: tensor of log-probabilites of actions of new policy (shape [n])
    :return: objective function for policy (shape [1])
    '''
    # computing the probability ratio
    logratio =  logprob - logprob_old
    ratio = logratio.exp()

    # clipping
    pg_loss1 =   advantages * ratio
    pg_loss2 =   advantages * torch.clamp(ratio, 1 - hypp.clip_coef, 1 + hypp.clip_coef)
    return torch.min(pg_loss1, pg_loss2).mean()

def compute_clipped_value_loss(returns, old_values, values,hypp):
    '''
    Applies the same idea of trust region optimization to the value function,
    constructing a clipped value loss.
    :param returns: batch of returns collected in the trajectory (shape [n])
    :param old_values: batch of value approximations of old policy (shape [n])
    :param values: batch of value approximations of updated policy (shape [n])
    :return the clipped value loss (shape [1])
    '''
    # TODO: Part b)
    v_loss_unclipped = (values - returns) **2
    v_clipped = old_values + torch.clamp(values - old_values, - hypp.clip_coef,hypp.clip_coef )
    v_loss_clipped = (v_clipped - returns) **2
    v_loss_max = torch.max(v_loss_unclipped,v_loss_clipped)
    v_loss = 0.5 * v_loss_max.mean()
    return v_loss