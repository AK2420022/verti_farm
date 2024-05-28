import numpy as np
import torch as torch
from typing import TYPE_CHECKING, Any, Callable, Dict, List, NamedTuple, Optional, Protocol, SupportsFloat, Tuple, Union
import copy
from gymnasium import spaces
from collections import OrderedDict

class ReplayBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor
    #achieved_goal: torch.Tensor
class HerBuffer():   
    def __init__(self,env,buffer_size,n_envs,observation_space,action_space,device,batch_size,goal_selection_strategy,n_goals=5):
        self.buffer_size = max(buffer_size//n_envs,1)
        self.observation_space = env.observation_space
        self.obs_shape =self.get_obs_shape(self.observation_space )
        self.action_dim = action_space.shape
        self.action_space = action_space
        self.n_envs = n_envs
        self.device = device
        self.n_goals = n_goals
        self.env = env
        self.goal_selection_strategy = goal_selection_strategy
        self.her_ratio = 1 - (1.0 / (n_goals + 1))
        self.batch_size = batch_size
        
        ###############initialize storage
        self.observations = OrderedDict()
        goal_ordered_dict = OrderedDict()

        for key, value in self.obs_shape['achieved_goal'].items():
            goal_ordered_dict[key] = np.zeros((value))
        self.goal_dtype = goal_ordered_dict
        self.observations['achieved_goal'] = np.full((self.buffer_size, self.n_envs), self.goal_dtype , dtype=object)
                             
        self.observations['desired_goal']  = np.full((self.buffer_size, self.n_envs), self.goal_dtype, dtype=object)
                       
        self.observations['observation'] = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape['observation']),dtype= np.float32)
        self.next_observations = OrderedDict()
        self.next_observations['achieved_goal'] = np.full((self.buffer_size, self.n_envs), self.goal_dtype, dtype=object)
        self.next_observations['desired_goal']  = np.full((self.buffer_size, self.n_envs), self.goal_dtype, dtype=object)
        self.next_observations['observation'] = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape['observation']), dtype=np.float32)


        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.action = np.zeros((self.buffer_size, self.n_envs, *self.action_dim), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_start = np.zeros((self.buffer_size, self.n_envs), dtype=np.int64)
        self.episode_length = np.zeros((self.buffer_size, self.n_envs), dtype=np.int64)
        self.current_episode_start = np.zeros(self.n_envs, dtype=np.int64)
        self.buffer_count = 0   
        self.infos = np.array([[{} for _ in range(self.n_envs)] for _ in range(self.buffer_size)])
        self.epsilon = 1e-6
        self.obs_clip = observation_space['observation'].low
        self.full = False

    def get_obs_shape(
        self,observation_space: spaces.Space
    ) -> Union[Tuple[int, ...], Dict[str, Tuple[int, ...]]]:
        """
        Get the shape of the observation (useful for the buffers).

        :param observation_space:
        :return:
        """
        if isinstance(observation_space, spaces.Box):
            return observation_space.shape
        elif isinstance(observation_space, spaces.Discrete):
            # Observation is an int
            return (1,)
        elif isinstance(observation_space, spaces.MultiDiscrete):
            # Number of discrete features
            return (int(len(observation_space.nvec)),)
        elif isinstance(observation_space, spaces.MultiBinary):
            # Number of binary features
            return observation_space.shape
        elif isinstance(observation_space, spaces.Dict):
            return {key: self.get_obs_shape(subspace) for (key, subspace) in observation_space.spaces.items()}  # type: ignore[misc]

        else:
            raise NotImplementedError(f"{observation_space} observation space is not supported")


    @staticmethod
    def _maybe_cast_dtype(dtype: np.typing.DTypeLike) -> np.typing.DTypeLike:
        """
        Cast `np.float64` action datatype to `np.float32`,
        keep the others dtype unchanged.
        See GH#1572 for more information.

        :param dtype: The original action space dtype
        :return: ``np.float32`` if the dtype was float64,
            the original dtype otherwise.
        """
        if dtype == np.float64:
            return np.float32
        return dtype
    def to_torch(self, array: np.ndarray, copy: bool = True) -> torch.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data (may be useful to avoid changing things
            by reference). This argument is inoperative if the device is not the CPU.
        :return:
        """
        if copy:
            return torch.tensor(array, device=self.device)
        return torch.as_tensor(array, device=self.device)

    def add(self,obs, next_obs, noisy_action, reward, done, infos):
        # The `episode_start` variable in the `HerBuffer` class is used to keep track of the
        # starting index of each episode in the replay buffer. It is an array that stores the
        # indices where each episode begins within the buffer. This information is important for
        # various operations within the `HerBuffer` class, such as calculating the length of
        # episodes, selecting additional goals based on different strategies, and handling
        # episode transitions.
        pos = self.buffer_count % self.buffer_size
        for env in range(self.n_envs):
            episode_start = self.episode_start[pos, env]
            episode_length = self.episode_length[pos, env]
            if episode_length > 0:
                episode_end = episode_start + episode_length
                episode_indices = np.arange(pos, episode_end) % self.buffer_size
                self.episode_length[episode_indices, env] = 0

            # Copy to avoid modification by reference
            for key,dicta in self.observations.items():
                # Reshape needed when using multiple envs with discrete observations
                # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)

                self.observations[key][pos][env] = obs[0][key]

            for key,dicta in self.next_observations.items():

                self.next_observations[key][pos][env] = next_obs[key]
          
            self.infos[pos][env] = infos
            self.episode_start[pos][env]  = self.current_episode_start
            noisy_action = noisy_action#.reshape((self.n_envs, *self.action_dim))
            
            self.action[pos][env]  = np.array(noisy_action)
            self.dones[pos][env]  = np.array(done)
            self.rewards[pos][env]  = np.array(reward)
        self.buffer_count += 1
        if self.buffer_count == self.buffer_size:
            self.full = True
            self.buffer_count = 0
        for env in range(self.n_envs):
            if done[env]:
                start = self.current_episode_start[env]
                episode_end = self.buffer_count 
                if episode_end < start:
                    episode_end += self.buffer_size
                indices = np.arange(start,episode_end)%self.buffer_size
                self.episode_length[indices,env] = episode_end - start
                self.current_episode_start[env] = self.buffer_count 
        

    def get_additional_goals(self, batch_indices,env_indices,goal_selection_strategy):
        # The `episode_start` variable in the `additional_goals` method is used to retrieve the
        # starting index of the episode corresponding to the given `batch_indices` and `env_id`. This
        # starting index is important for determining the position of the goals within the episode. It
        # helps in calculating the correct indices for selecting additional goals based on the
        # specified `goal_selection` strategy.
        
        episode_start = self.episode_start[batch_indices,env_indices]
        episode_length = self.episode_length[batch_indices,env_indices]

        if goal_selection_strategy == "final":
            goals_indices = episode_length - 1
            
        elif goal_selection_strategy == "future":
            current_episode = (batch_indices  - episode_start) % self.buffer_size
            # The code `goals_indices` is not valid Python syntax. It seems like it might be a
            # variable name or a function call, but without more context or code, it is difficult to
            # determine its purpose.
            goals_indices = np.random.randint(current_episode,self.buffer_count)
            #print("episode start", episode_start.flatten())
        elif goal_selection_strategy == "episode":
            goals_indices = np.random.randint(0, episode_length,size=1)
        else :
            print("wrong goal selection strategy")
        indices = (episode_start+ goals_indices) % self.buffer_size
        additional_goals = self.next_observations["achieved_goal"][indices]
        # print("next")
        # print(self.next_observations["achieved_goal"][indices])  
        # print("neaxt")
        # print(self.next_observations["achieved_goal"])  
        #print("add gaols: ",additional_goals[0])
        
        return additional_goals

    
    def sample(self,batch_size):
        ###############initialize storage
        obs_sample = OrderedDict()
        obs_sample['achieved_goal'] = np.full((self.batch_size, self.n_envs), self.goal_dtype , dtype=object)
                             
        obs_sample['desired_goal']  = np.full((self.batch_size, self.n_envs), self.goal_dtype, dtype=object)
                       
        obs_sample['observation'] = np.zeros((self.batch_size, self.n_envs, *self.obs_shape['observation']),dtype= np.float32)
        next_obs_sample = OrderedDict()
        next_obs_sample['achieved_goal'] = np.full((self.batch_size, self.n_envs), self.goal_dtype, dtype=object)
        next_obs_sample['desired_goal']  = np.full((self.batch_size, self.n_envs), self.goal_dtype, dtype=object)
        next_obs_sample['observation'] = np.zeros((self.batch_size, self.n_envs, *self.obs_shape['observation']), dtype=np.float32)
        #################################
        batch_indices = np.random.randint(0,self.buffer_count-1,size = int(batch_size))
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_indices)))
        # The code snippet is splitting a list of batch indices into two parts based on a certain
        # ratio (`self.her_ratio`). The first part (`real_indices`) contains indices from the
        # beginning up to a certain point determined by the ratio, and the second part
        # (`virtual_indices`) contains the remaining indices from that point to the end of the list.
        real_indices, virtual_indices = batch_indices[:int(batch_size * self.her_ratio)], batch_indices[int(batch_size * self.her_ratio):]
        real_env_indices, virtual_env_indices =env_indices[:int(batch_size * self.her_ratio)], env_indices[int(batch_size * self.her_ratio):]
        real_actions = torch.tensor(self.action[real_indices, real_env_indices])
        real_dones = torch.tensor(self.dones[real_indices, real_env_indices]).reshape(-1, 1)
        real_rewards = torch.tensor(self.rewards[real_indices, real_env_indices]).reshape(-1, 1)
        virtual_actions = torch.tensor(self.action[virtual_indices, virtual_env_indices])
        virtual_rewards = np.zeros((real_rewards.shape))
        obs_ind = np.arange(0,len(real_indices))
        next_obs_ind = np.arange(len(real_indices),len(real_indices)+len(virtual_indices))
        for env in range(self.n_envs):
            for key,dicta in self.observations.items():
                obs_sample[key][obs_ind][env] = self.observations[key][real_indices][env]

            for key,dicta in self.next_observations.items():
                next_obs_sample[key][obs_ind][env]= self.next_observations[key][real_indices][env]       
            # `virtual_observations` is a dictionary containing the observations for the virtual samples
            # in the HER (Hindsight Experience Replay) algorithm. These virtual samples are generated to
            # improve the learning process by replaying experiences with different goals.
            for key,dicta in self.observations.items():
                obs_sample[key][next_obs_ind][env] = self.observations[key][virtual_indices][env]
            for key,dicta in self.next_observations.items():
                next_obs_sample[key][next_obs_ind][env] = self.next_observations[key][virtual_indices][env]
            
            goals = self.get_additional_goals(next_obs_ind,virtual_env_indices,self.goal_selection_strategy)
        next_obs_sample["desired_goal"][next_obs_ind] = goals
        obs_sample["desired_goal"][next_obs_ind] = goals
        virtual_infos = copy.deepcopy(self.infos[virtual_indices, virtual_env_indices])


        achieved_goal_sample = next_obs_sample["achieved_goal"][next_obs_ind]
        desired_goal_sample = obs_sample["desired_goal"][next_obs_ind]        
        virtual_rewards = np.zeros((len(virtual_indices),self.n_envs))
        #print("Desired goal: ", desired_goal_sample)
        #print("achieved goal: ", achieved_goal_sample)
        for ind in range(len(virtual_indices)):
            #print("Desired goal: ", achieved_goal_sample[ind][0])
            virtual_rewards[ind] =1 * self.env.envs[0].compute_reward(
                achieved_goal_sample[ind][0],
                # here we use the new desired goal
                desired_goal_sample[ind][0],
                virtual_infos
            )
        virtual_dones = torch.tensor(self.dones[virtual_indices, virtual_env_indices]).reshape(-1, 1)
        virtual_rewards = torch.tensor(virtual_rewards)


        actions = torch.cat((real_actions, virtual_actions))
        dones = torch.cat((real_dones, virtual_dones))

        rewards = torch.cat((real_rewards, virtual_rewards))
        #make it hypp 
        rewards = rewards
        data = (obs_sample,actions,next_obs_sample,dones,rewards)

        return  ReplayBufferSamples(*tuple(data))





        