
import torch
import random
import numpy as np
import copy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    '''
    returns the linearly decreasing exploration probability epsilon for a certain timestep
    :param start_e: exploration probability (epsilon) at timestep 0
    :param end_e:  minimal exploration probability (epsilon)
    :param duration: number of timesteps after which end_e is reached
    :param t: the current timestep
    :return: exploration probability (epsilon) for the current timestep
    '''
    slope = (end_e - start_e)  / duration
    return max(slope * t + start_e, end_e)

def compute_TD_target(data, agent, gamma):
    '''
    returns the TD-targets for the given data batch.
    :param data: batch of transitions as named tuple of tensors (observations, actions, next_observations, dones, rewards)
    :param agent: an Agent class object
    :return: TD-targets as tensor with shape [batch_size], where batch_size is the number of entries in data tuple
    '''
    with torch.no_grad():
        target_max, _ = agent.get_q_values(data.next_observations).max(dim=1)
        td_target = data.rewards.flatten() +gamma * target_max * (1 - data.dones.flatten())

    return td_target

def e_greedy_policy(agent, obs, epsilon, env):
    '''
    returns the action following the epsilon-greedy policy.
    :param agent: Agent class object
    :param obs: observation of the state
    :epsilon: probability to choose a random action
    :env: environment agent is acting in
    '''
    if random.random() < epsilon:
        action = np.array([env.single_action_space.sample()])
    else:
        q_values = agent.get_target_values(torch.Tensor(obs).to(device))
        action = torch.argmax(q_values, dim=1).cpu().numpy()

    return action

def compute_TD_target_DDQN(data, agent,gamma):
    '''
    returns the TD-targets for the given data batch according to the Double-DQN (DDQN) paper.
    :param data: batch of transitions as named tuple of tensors (observations, actions, next_observations, dones, rewards)
    :param agent: an Agent class object
    :return: TD-targets as tensor with shape [batch_size], where batch_size is the number of entries in data tuple
    '''
    
    with torch.no_grad():
      
        action = agent.get_action(torch.Tensor(data.next_observations).to(device))  
        target_max= agent.get_target_values(torch.Tensor(data.next_observations).to(device))[:,action]
        
        td_target = data.rewards.flatten() + gamma * target_max * (1 - data.dones.flatten())

    return td_target

