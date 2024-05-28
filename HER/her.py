from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv
from enum import Enum
import numpy as np
import torch
import copy
from typing import  Dict,  NamedTuple
class DictReplayBufferSamples(NamedTuple):
    observations: Dict[str, torch.Tensor]
    actions: torch.Tensor
    next_observations: Dict[str, torch.Tensor]
    dones: torch.Tensor
    rewards: torch.Tensor

class HerBuffer():
    def __init__(self,env:VecEnv, buffer_size:int,observation_space:spaces.Dict,action_space:spaces.Space,device, goal_selection_strategy:str = "final"):

        self.env = env 
        self.action_space= action_space
        self.goal_selection_strategy = goal_selection_strategy
        self.default_selection_strategy = ["final", "episode", "random"]
        assert (self.goal_selection_strategy in self.default_selection_strategy),f"invalid selection strategy, use one of {self.default_selection_strategy}"
        # In some environments, the info dict is used to compute the reward. Then, we need to store it.
        self.num_envs = env.num_envs
        self.infos = np.array([[{} for _ in range(self.num_envs)] for _ in range(buffer_size)])

        #instantiate buffer 
        self.observation_space = observation_space
        self.actions_space = action_space
        self.observation_shape = env.observation_space.shape
        self.action_shape = np.array(action_space.shape).prod()
        self.buffer_size = buffer_size/self.num_envs 
        self.observations = np.zeros(
            (self.buffer_size, self.num_envs, self.observation_shape), dtype=self._maybe_cast_dtype(action_space.dtype)
        )
        self.next_observations = np.zeros(
            (self.buffer_size, self.num_envs, self.observation_shape), dtype=self._maybe_cast_dtype(action_space.dtype)
        )
        self.actions = np.zeros(
            (self.buffer_size, self.num_envs, self.action_shape), dtype=self._maybe_cast_dtype(action_space.dtype)
        )
        self.rewards = np.zeros((self.buffer_size, self.num_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.num_envs), dtype=np.float32)
        self.buffer_level = 0
        self.current_start = np.zeros((self.num_envs, 1))
        self.ep_start = np.zeros((self.buffer_size, self.num_envs), dtype=np.int64)
        self.ep_length = np.zeros((self.buffer_size, self.num_envs), dtype=np.int64)
        #todo handle timeout termination

    def add(self,obs,next_obs,action,reward,done,infos):
        for env_id in range(self.num_envs):
            start = self.ep_start[self.buffer_level,env_id]
            length = self.ep_end[self.buffer_level,env_id]
            episode_indices = np.arange(self.buffer_level,start + length)
            self.ep_length[episode_indices,env_id] = 0
        
        self.ep_start[self.buffer_level] = self.current_start
        #handle discrete spaces. mentioned in stable_baselines implementation
        for key, observation in obs.items():
            # Check if observation is discrete
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                # Reshape discrete observations for multiple environments
                reshaped_observation = observation.reshape((self.num_envs,) + self.obs_shape[key])
                # Store reshaped observation
                self.observations[self.buffer_level] = np.array(reshaped_observation)
            else:
                # Store observation directly
                self.observations[self.buffer_level] = np.array(observation)

        for key, observation in next_obs.items():
            # Check if observation is discrete
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                # Reshape discrete observations for multiple environments
                reshaped_observation = observation.reshape((self.num_envs,) + self.obs_shape[key])
                # Store reshaped observation
                self.observations[self.buffer_level] = np.array(reshaped_observation)
            else:
                # Store observation directly
                self.observations[self.buffer_level] = np.array(observation)

        action = action.reshape((self.num_envs,self.action_shape))
        self.actions[self.buffer_level] = np.array(action)
        self.rewards[self.buffer_level] = np.array(reward)
        self.dones[self.buffer_level] = np.array(done)
        self.infos[self.buffer_level] = np.array(infos)
        self.buffer_level += 1
        if self.buffer_level >= self.buffer_size:
            self.buffer_level = 0
        for env_id in range(self.num_envs):
            if done[env_id]:
                start = self.current_start[env_id]
                end = self.buffer_level 
                if end  < start:
                    end += self.buffer_size
                episode_indices = np.arange(start, end) % self.buffer_size
                self.ep_length[episode_indices, env_id] = end - start
                self.current_start[env_id] = self.buffer_level
        
    def sample(self,batch_size: int ,k:float):
        is_valid = self.ep_length > 0
        valid_indices = np.flatnonzero(is_valid)
        sampled_indices = np.random.choice(valid_indices, size=batch_size, replace=True)
        batch_indices, env_indices = np.unravel_index(sampled_indices, is_valid.shape)
        additional_batch, actual_batch = np.split(batch_indices,int(k))
        additional_env, actual_env = np.split(env_indices,int(k))
        #actual samples
        observations_actual = torch.tensor(self.observations[actual_batch, actual_env, :])
        next_observations_actual = torch.tensor(self.next_observations[additional_batch, additional_env, :]) 
        dones_actual = torch.tensor(self.dones[batch_indices,env_indices]).reshape(-1,1)
        rewards_actual = torch.tensor(self.rewards[batch_indices,env_indices]).reshape(-1,1)
        #additional samples
        new_goals = self.sample_goals(batch_indices, env_indices)
        observations_additional = torch.tensor(self.observations[actual_batch, actual_env, :]) 
        observations_additional["desired_goal"] = new_goals
        next_observations_additional =  torch.tensor(obs[additional_batch, additional_env, :]) for key, obs in self.next_observations.items()}
        next_observations_additional["desired_goal"] = new_goals
        infos = copy.deepcopy(self.infos[batch_indices, env_indices])
        actions_additional = torch.tensor(self.actions[batch_indices, env_indices])
        dones_additional = torch.tensor(self.dones[batch_indices,env_indices]).reshape(-1,1)
        #todo design this reward function
        rewards = self.env.env_method("compute_reward",next_observations_additional["achived_goal"],observations_additional["desired_goal"],infos,indices=[0])
        rewards_additional = rewards[0]
        # Concatenate real and virtual data
        observations = {
            torch.cat((observations_actual[key], observations_additional[key]))
        }
        actions = torch.cat((actions_actual, actions_additional))
        next_observations = {
            key: torch.cat((next_observations_actual[key],next_observations_additional[key]))
            for key in next_observations_additional.keys()
        }
        dones = torch.cat((dones_actual, dones_additional))
        rewards = torch.cat((rewards_actual,rewards_additional))       
        return DictReplayBufferSamples(
            observations=observations,
            actions=actions,
            next_observations=next_observations,
            dones=dones,
            rewards=rewards,
        )
    def sample_goals(self, batch_indices: np.ndarray, env_indices: np.ndarray):
        start = self.ep_start[batch_indices, env_indices]
        length = self.ep_length[batch_indices, env_indices]
        if self.goal_selection_strategy == "final":
            goals_indices = length-1
        elif self.goal_selection_strategy == "future":
            current_indices = (batch_indices  - start) % self.buffer_size
            goals_indices = np.random.randint(current_indices,length)
        elif self.goal_selection_strategy == "episode":
            goal_indices = np.random.randint(0,length)
        
        goals_indices = (goals_indices+start)%self.buffer_size
        return self.next_observations["achived_goal"][goal_indices,env_indices]
    