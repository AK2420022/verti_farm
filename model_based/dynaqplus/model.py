import numpy as np
from collections import OrderedDict
from gym import spaces
class Model(object):
    def __init__(self, observation_space,action_space,alpha,gamma,n,eps,model_learning_rate):
        self.action_space = action_space
        self.observation_space = observation_space
        self.alpha = alpha
        self.gamma = gamma
        self.n = n
        self.eps = eps
        self.model_learning_rate = model_learning_rate
        if  isinstance(observation_space, spaces.Discrete) or isinstance(observation_space, spaces.MultiDiscrete):
            
            self.obs_shape = observation_space.nvec[0]
        else:
            self.obs_shape = self.observation_space["observation"].n if isinstance(observation_space, spaces.Dict) else np.array(self.observation_space).prod()    
        self.states = {}
        if  isinstance(action_space, spaces.Discrete) or isinstance(action_space, spaces.MultiDiscrete):
            self.action_shape = action_space.nvec[0]
        else:
            self.action_shape = np.array(self.action_space.shape).prod()
        self.actions = {}
        self.q = np.zeros((self.obs_shape,self.action_shape))
        self.rewards = np.zeros((self.obs_shape,self.action_shape))
        self.model = np.zeros((self.obs_shape, self.action_shape, 3))
    def get_action(self,observation):
        if np.random.rand() < self.eps:
            action = np.random.choice(range(self.action_shape))
        else:
            action = np.random.choice(np.flatnonzero(self.q[observation]==self.q[observation].max()))
        if isinstance(self.action_space, spaces.Discrete) or isinstance(self.action_space, spaces.MultiDiscrete):
            
            action = np.array(int(action)).reshape(-1,1)[0]

        return action
    
    def add_transition(self,observation,next_observation,action,reward,elapsed_time):
        for action in action:    
            for obs in observation:
                    if obs not in self.states.keys():
                        self.states[str(obs)] = 1
                        self.actions[str(obs)] = [action]
                    
                    else:
                        self.states[str(obs)] +=1
                        if action not in self.actions[str(obs)]:
                            self.actions[str(obs)].add(action)
                    self.rewards[obs][action] = reward 
                    self.model[obs][action] =  np.array([next_observation, reward, elapsed_time])
    def q_update(self,dynaplus,time_weight,current_time):
        
        for i in range(self.n):
            state =  np.random.choice(list(self.states))
            if len(self.actions[state]) <=0:
                action = np.random.randint(self.action_shape)
            else:
                action = np.random.choice(self.actions[state])
            next_obs , reward, elapsed_time = self.model[int(state)][int(action)]
            if dynaplus:
                reward +=  time_weight * np.sqrt(current_time - elapsed_time)
            self.compute_q([state], [action],next_obs,reward)

    def compute_q(self, state, action,next_obs,reward):
        if  isinstance(self.action_space, spaces.Discrete) or isinstance(self.action_space, spaces.MultiDiscrete):
            action = int(action[0])
        if  isinstance(self.observation_space, spaces.Discrete) or isinstance(self.observation_space, spaces.MultiDiscrete):    
            next_obs = int(next_obs)
            state = int(state[0])
        self.q[state][action] += self.alpha * (reward + self.gamma * max(self.q[next_obs]) - self.q[state][action])
    
    #def get_action_discrete(self, action):
        