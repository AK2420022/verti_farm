import os
import copy
import time
import random
import warnings
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from tqdm import notebook
from easydict import EasyDict as edict
from IPython.display import Video
from ensemble import Ensemble
import utils.helper_fns as hf
from tqdm import tqdm
import gymnasium as gym
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer, ReplayBufferSamples
from moviepy.editor import *
import cv2
import time
from Actor import Actor
from Critic import Critic#,Critic2
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
warnings.filterwarnings("ignore", category=DeprecationWarning)

os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['WANDB_NOTEBOOK_NAME'] = 'dqn.py'

plt.rcParams['figure.dpi'] = 100
device = torch.device( "cpu")
class mbpo():
    """
    The mbpo class implements the Model-Based Policy Optimization (MBPO) algorithm.
    """
    def __init__(self,exp =edict(),hypp=edict()):
        """
        Initializes the MBPO class with the given experiment and hyperparameter configurations.

        :param exp: Experiment configuration. Default is an empty edict.

        :param hypp: Hyperparameter configuration. Default is an empty edict.
        """
        if exp.set_random_seed:
            random.seed(exp.seed)
            np.random.seed(exp.seed)
            torch.manual_seed(exp.seed)
            torch.backends.cudnn.deterministic = exp.set_random_seed

        self.env = gym.vector.SyncVectorEnv([hf.make_env(exp.env_id, exp.seed + i) for i in range(1)])
        self.env_rb = ReplayBuffer(
            hypp.buffer_size,
            self.env.observation_space,
            self.env.action_space,
            device,
            handle_timeout_termination = False,
        )
        self.model_rb = ReplayBuffer(
            hypp.buffer_size,
            self.env.observation_space,
            self.env.action_space,
            device,
            handle_timeout_termination = False,
        )
        self.env.close()
        self.run_name = f"{exp.env_id}__{exp.exp_name}__{exp.seed}__{datetime.now().strftime('%y%m%d_%H%M%S')}"
        if exp is not None:
            self.exp = exp
        if hypp is not None:
            self.hypp = hypp
        # Init tensorboard logging and wandb logging
        self.writer = hf.setup_logging(f"{exp.env_id}", exp, hypp)
        self._rollout_length = self.hypp.num_rollouts
        self._rollout_schedule = self.hypp.rollout_schedule
        self.angle_threshold = torch.full(size=(self.hypp.num_rollouts, 1,1),fill_value=0.2)
        self.action_penalty = 0.8
        self.directional_reward = torch.full(fill_value=1,size=(self.hypp.num_rollouts, 1,1))
        self.prev_pole_angle = torch.zeros(size=(self.hypp.num_rollouts, 1,1))
        self.prev_action = torch.zeros(size=(self.hypp.num_rollouts, 1,1))
    def scale(self, reward,mean = 0.0,var = 1.0,count = 1e-10 ):
        """
        Normalizes the input reward using the running mean and variance.
        
        :param reward: The reward to be normalized.
        :return: The normalized reward.
        
        """
        count += 1
        mean = mean + (reward - mean) / count
        var = var + (reward - mean) * (reward - mean)
        std = (var / count) ** 0.5
        return (reward - mean) / std
    def generate_data(self,model_env,env_rb,samples = 5000):
        """
        Generates data using the model environment and stores it in the replay buffer.

        :param model_env: The model environment used for data generation.
        :param env_rb: The replay buffer where generated data is stored.
        :param samples: The number of samples to generate. Default is 5000.
        """
        obs = model_env.reset()[0]
        episode_step = 0
        for _ in range(samples):
            action = model_env.action_space.sample()
            next_obs, reward,done,truncated, info = model_env.step(action)
            done =np.logical_or( done ,  truncated)
            
            env_rb.add(obs, next_obs, action, reward, done, info)
            obs = next_obs
            episode_step += 1
            if done or episode_step %1000 == 0:
                obs = model_env.reset()[0]

    def do_rollouts(self,env,policy,ensemble, env_buffer,model_buffer,batch_size):
        """
        Performs rollouts using the given policy and ensemble, storing the results in the model buffer.

        :param env: The environment used for sampling initial states.
        :param policy: The policy used to generate actions.
        :param ensemble: The ensemble model used for predicting next states and rewards.
        :param env_buffer: The environment replay buffer for sampling initial states.
        :param model_buffer: The model replay buffer where rollout data is stored.
        :param batch_size: The number of samples to use for rollouts.

        """
        episode_step = 0
        states = env_buffer.sample(batch_size).observations
        accum_dones = torch.zeros(size=(batch_size,1), dtype=bool)

        for _ in range(self._rollout_length):
            #sample st from Denv
            action, log_prob = policy.get_action(states)
            next_obs, reward = ensemble.sample_predictions(states,action)
            #reward function (optional reward?)
            reward,done = self.compute_reward(env,next_obs,action,reward)
            not_done_mask = ~accum_dones
            truncated = False if episode_step < 1000 else True
            done = np.logical_or( done ,  truncated)
            self.insert_batch(model_buffer,states[not_done_mask],next_obs[not_done_mask],action[not_done_mask],reward[not_done_mask],done[not_done_mask],{})
            
            states = next_obs
            accum_dones =(accum_dones | done).to(torch.bool)
            episode_step +=1
            # Reset states if necessary
            # Break if all rollouts are done
            if done.all():
                break
    def insert_batch(self,model_buffer,obs,next_obs,action,reward,done,info):
        """
        Inserts a batch of transitions into the model buffer.

        :param model_buffer: The buffer where transitions are stored.
        :param obs: The current observations.
        :param next_obs: The next observations after taking the action.
        :param action: The actions taken from the current observations.
        :param reward: The rewards received after taking the actions.
        :param done: The done flags indicating whether the episode has ended.
        :param info: Additional information about the transitions.

        """
        for i in range(obs.shape[0]):
            model_buffer.add(obs[i],next_obs[i],action[i].detach().numpy(),reward[i],done[i],{})

    def compute_reward(self,env,state,action,reward):
        """
        Computes the reward based on the environment and the current state and action.

        :param env: The environment from which the reward is computed.
        :param state: The current state of the environment.
        :param action: The action taken from the current state.
        :return: A tuple (reward, done) where reward is the computed reward and done is a flag indicating
                whether the episode has ended.

        This method calculates the reward and done flag differently depending on the environment ID:
        
        - "HalfCheetah-v5": Computes the reward based on the x-velocity and control cost. 
        Penalizes negative forward rewards.
        - "InvertedPendulum-v5": Computes the reward based on the pole angle and action magnitude.
        Applies penalties or rewards based on changes in pole angle and action direction. Done flag is
        set if the pole angle exceeds a threshold.
        """
        if self.exp.env_id == "HalfCheetah-v5":
            x_vel = state[:,8]
            rewards = env.call("_get_rew",x_vel,action) 
            reward = rewards[0][0] 

            if rewards[0][1]["reward_forward"][0] < 0 :
                reward = rewards[0][1]["reward_forward"][0] * 0.8 - rewards[0][1]["reward_ctrl"] * 0.5
            done = False 
            return reward, done
        elif self.exp.env_id == "InvertedPendulum-v5":
            pole_angle = state[:,:,1]
            #absolute_change = old_angle - pole_angle
            # Penalize based on the pole angle
            #print("pole angle, ",pole_angle)
            angle_penalty = pole_angle ** 2
            angle_penalty = angle_penalty[:,:,np.newaxis]
            pole_angle = pole_angle[:,:,np.newaxis]
            # Penalize based on the action magnitude
            action_penalty = self.action_penalty * (action ** 2)
            #print("angle_penalty, ",angle_penalty)
            # Initialize the reward
            reward -=  (-angle_penalty - action_penalty)
            i=0
            for pole_ang, prev_angle,act,prev_act,angle_threshold in zip(pole_angle,self.prev_pole_angle,action.detach().numpy(),self.prev_action.detach().numpy(),self.angle_threshold):
                # Check if the pole angle increased and the action is in the same direction as the previous action
                if abs(pole_ang) > abs(prev_angle) and np.sign(act) == np.sign(prev_act):
                    # Penalize if the pole angle increased and action is in the same direction
                    reward[i] = -1#reward[i] -= self.directional_reward[i] 
                elif abs(pole_ang) > abs(prev_angle) and np.sign(act) != np.sign(prev_act):
                    # Reward if the pole angle increased and action is in the opposite direction
                    reward[i] = 2#reward[i] += self.directional_reward[i]
                #print("new reward: " , reward)
                # If the pole has fallen over (angle exceeds threshold), return a large negative reward
                
                if abs(pole_ang) > angle_threshold:
                    reward[i] = -2  # Large negative reward for falling over
                #print("new reward:, ",reward[i])
                i+=1
            # Update the previous angle and action
            self.prev_pole_angle = pole_angle
            self.prev_action = action
            #reward = 1.0 if np.abs(pole_angle) < 0.2 else -1
            done = (abs(pole_angle) > self.angle_threshold)
            #old_angle = pole_angle
            return reward.detach().numpy(), done
    def print_param_values(self,model):
        """
        Prints the values of the parameters of the given model.

        :param model: The model whose parameters are to be printed.

        This method iterates over the named parameters of the model and prints the name and value of each parameter.
        """
        for name, param in model.named_parameters():
            print(f"Parameter {name} value: {param.data}")

    def linear_scheduler(self,epoch):
        """
        Returns the linear decay value for the learning rate scheduler.

        :param epoch: The current epoch number.
        :return: The decay value based on the current epoch and total timesteps.

        This method computes a linear decay value that decreases from 1 to 0 over the total number of timesteps.
        """
        return 1 - epoch / self.hypp.total_timesteps

    def _set_rollout_length(self,episode_step):
        """
        Sets the rollout length based on the current episode step.

        :param episode_step: The current step of the episode.

        This method adjusts the rollout length according to a predefined schedule. The rollout length increases
        linearly from a minimum to a maximum value based on the episode step.
        """
        min_epoch, max_epoch, min_length, max_length = self._rollout_schedule
        if episode_step <= min_epoch:
            y = min_length
        else:
            dx = (episode_step - min_epoch) / (max_epoch - min_epoch)
            dx = min(dx, 1)
            y = dx * (max_length - min_length) + min_length
        self._rollout_length = int(y)
        #print("_rollout_length, ",self._rollout_length)
    def train(self):
        """
        Trains the model and policy.

        This method performs the training loop, including environment interactions, policy updates, and model updates.
        It evaluates the agent's performance periodically and logs training metrics.
        """
        # create two vectorized envs: one to fill the rollout buffer with trajectories and
        # another one to evaluate the agent performance at different stages of training
        # Note: vectorized environments reset automatically once the episode is finished
        env = gym.vector.SyncVectorEnv([hf.make_env(self.exp.env_id, self.exp.seed)])
        env_eval = gym.vector.SyncVectorEnv([hf.make_env(self.exp.env_id, self.exp.seed + i) for i in range(1)])
        ###################################
        #ent_coef init
        log_entropy = - np.array(env.action_space.shape).prod()
        log_alpha = torch.tensor([1.0], requires_grad=True)
        ent_coef = 0.4
        alpha_optimizer = optim.Adam(params=[log_alpha], lr=self.hypp.learning_rate) 
        scheduler_alpha = torch.optim.lr_scheduler.LambdaLR(alpha_optimizer, lr_lambda=self.linear_scheduler)

        ###############################
        #initialize mdeol
        model_env = Ensemble(env,self.hypp.learning_rate_model,num_ensembles = self.hypp.num_ensembles)
        #model_env.init_weights()
        ###################################################################
        # Create Actor class Instance and network optimizer
        policy = Actor(env,self.hypp.hidden_dim,self.hypp.hidden_layers_actor).to(device)
        policy.init_weights()
        optimizer_actor = optim.Adam(policy.parameters(), lr=self.hypp.learning_rate_actor )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer_actor, lr_lambda=self.linear_scheduler)
        #scheduler = lr_scheduler.LinearLR(optimizer_actor, start_factor=1.0, end_factor=0.3, total_iters=5000)
        #####################################################################
        # Create Critic class Instance and network optimizer Q1
        critic = Critic(env,self.hypp.hidden_dim ,self.hypp.hidden_layers_critic).to(device)
        critic.init_weights()
        critic_target = copy.deepcopy(critic)
        optimizer_q1 = optim.Adam(critic.q1.parameters() ,lr=self.hypp.learning_rate_critic )
        optimizer_q2 = optim.Adam(critic.q2.parameters(),lr=self.hypp.learning_rate_critic )
        scheduler_q1 = torch.optim.lr_scheduler.LambdaLR(optimizer_q1, lr_lambda=self.linear_scheduler)
        scheduler_q2 = torch.optim.lr_scheduler.LambdaLR(optimizer_q2, lr_lambda=self.linear_scheduler)

        # Create Critic class Instance and network optimizer Q2
        # init list to track agent's performance throughout training
        tracked_returns_over_training = []
        tracked_episode_len_over_training = []
        tracked_episode_count = []
        last_evaluated_episode = None  # stores the episode_step of when the agent's performance was last evaluated
        eval_max_return = -float('inf')
        model_loss  = 0
        # Init observation to start learning
        start_time = time.time()
        elapsed_time = 0
        obs = env.reset() 
        obs = obs[0]
        global_reward = 0
        global_step = 0
        episode_step = 0
        gradient_step = 0
        steps = tqdm(range(1, self.hypp.total_timesteps + 1))
        self.generate_data(env,self.env_rb,samples = 5000)
        episode_reward = 0
        episode_len = 0
        # training loop
        for update in steps:
            #take action and add to Denv
            data_env = self.env_rb.sample(self.hypp.batch_size)
            action,log_prob = policy.get_action(obs)
            # apply action to environment
            next_obs, reward, done,truncated, infos = env.step(action.detach().numpy())
            done = done or  truncated or (update%1000 ==0)
            
            global_step += 1
            global_reward += reward
            episode_reward += reward
            #print("episode_reward, ", infos)
            episode_len +=1
            self.env_rb.add(obs,next_obs,action.detach().numpy(),reward,done,infos)
            if not done:
                obs = next_obs
            if done:
                obs = env.reset()[0]
            # log episode return and length to tensorboard as well as current epsilon
            for info,values in infos.items():
                if "episode" == info:
                    print("final_info")
                    print("values: ", values)
                    
                    #print("global step: ", global_step)
                    episode_step += 1
                    steps.set_description(f"global_step: {global_step}, episodic_return={values['r']}")
                    self.writer.add_scalar("rollout/episodic_return", values["r"], global_step)
                    self.writer.add_scalar("rollout/episodic_length", values["l"], global_step)
                    self.writer.add_scalar("Charts/episode_step", episode_step, global_step)
                    self.writer.add_scalar("Charts/gradient_step", gradient_step, global_step)
                    break

            # evaluation of the agent
            if self.exp.eval_agent and (global_step % self.exp.eval_frequency == 0) and last_evaluated_episode != episode_step:
                last_evaluated_episode = episode_step
                tracked_return, tracked_episode_len = hf.evaluate_agent(env_eval, policy, self.exp.eval_count,
                                                                        self.exp.seed, greedy_actor=True)
                tracked_returns_over_training.append(tracked_return)
                tracked_episode_len_over_training.append(tracked_episode_len)
                tracked_episode_count.append([episode_step, global_step])
                # if there has been improvement of the model - save model, create video, log video to wandb
                if np.mean(tracked_return) > eval_max_return:
                    eval_max_return = np.mean(tracked_return)
                    # call helper function save_and_log_agent to save model, create video, log video to wandb
                    hf.save_and_log_agent(self.exp, policy, episode_step,
                                        greedy=True, print_path=False)


            # handling the terminal observation (vectorized env would skip terminal state)
            # log td_loss and q_values to tensorboard
            if global_step % 100 == 0:
                self.writer.add_scalar("others/SPS", int(global_step / (time.time() - start_time)), global_step)
                self.writer.add_scalar("Charts/episode_step", episode_step, global_step)
                self.writer.add_scalar("Charts/gradient_step", gradient_step, global_step)            
            
            if  global_step > self.hypp.rollout_start:
                if global_step % self.hypp.train_frequency == 0:
                    self._set_rollout_length(episode_step)
            #dp model rollouts
                    self.do_rollouts(env,policy,model_env,self.env_rb,self.model_rb,self.hypp.num_rollouts)
            #update grads
            if global_step > self.hypp.start_learning:
                if global_step % self.hypp.model_train_frequency == 0:
                    model_loss, trained_epochs = model_env.train(data_env)
                if global_step % self.hypp.train_frequency == 0:

                        
                    update_param = False
                    if global_step % self.hypp.update_param_frequency == 0:
                        update_param = True
                        gradient_step+=1
                    ########################################################3
                    # Hyperparameters and ratios
                    max_grad_norm = 2
                    ratio = self.hypp.real_img_ratio
                    real_size = int(self.hypp.batch_size * ratio)
                    img_data_size = int(self.hypp.batch_size * (1 - ratio))
                    
                    # Sample data
                    real_data = self.env_rb.sample(real_size)
                    img_data = self.model_rb.sample(img_data_size)
                    data = ReplayBufferSamples(
                        observations=torch.cat((real_data.observations, img_data.observations), dim=0),
                        actions=torch.cat((real_data.actions, img_data.actions), dim=0),
                        next_observations=torch.cat((real_data.next_observations, img_data.next_observations), dim=0),
                        dones=torch.cat((real_data.dones, img_data.dones), dim=0),
                        rewards=torch.cat((real_data.rewards, img_data.rewards), dim=0)
                    )
                    # Reward scaling
                    rewards =data.rewards# self.scale(data.rewards)
                    #print(rewards)
                    # Current action and probabilities
                    actions, log_prob = policy.get_action(data.observations.squeeze(1).to(torch.float32), deterministic=False)
                    log_prob_ = log_prob.clone().detach()
                    # Update entropy coefficient
                    ent_coef_loss = -torch.mean((log_alpha * (log_entropy + log_prob_)))
                    alpha_optimizer.zero_grad()
                    ent_coef_loss.backward()
                    grad_norm = log_alpha.grad.abs()
                    if grad_norm > max_grad_norm:
                        log_alpha.grad *= max_grad_norm / grad_norm
                    alpha_optimizer.step()
                    scheduler_alpha.step()
                    ent_coef = torch.exp(log_alpha)

                    # Update actor
                    q1_actor, q2_actor = critic(data.observations.squeeze(1), actions)
                    min_q_actor = torch.min(q1_actor, q2_actor)
                    policy_loss = torch.mean((ent_coef * log_prob) - min_q_actor)
                    #print("before")
                    #self.print_param_values(policy)
                    optimizer_actor.zero_grad()
                    policy_loss.backward()
                    clip_grad_norm_(policy.parameters(), max_grad_norm)
                    optimizer_actor.step()
  
                    #print("after")
                    self.print_param_values(policy)
                    # Update critic
                    with torch.no_grad():
                        new_action, new_log_prob = policy.get_action(data.next_observations.to(torch.float32).squeeze(1), deterministic=False)
                        new_q_value1, new_q_value2 = critic_target(data.next_observations.squeeze(1), new_action)
                        new_q_value_min = torch.min(new_q_value1, new_q_value2)
                        new_q_value_target = new_q_value_min - ent_coef * new_log_prob
                        target_q_value = rewards + self.hypp.gamma * new_q_value_target * (1 - data.dones)
                    
                    q1, q2 = critic(data.observations.squeeze(1), data.actions)
                    q_value_loss = (F.mse_loss(q1, target_q_value) + F.mse_loss(q2, target_q_value)).mean()
                    optimizer_q1.zero_grad()
                    optimizer_q2.zero_grad()
                    q_value_loss.backward()
                    self.print_param_values(critic)
                    clip_grad_norm_(critic.parameters(), max_grad_norm)
                    optimizer_q1.step()
                    optimizer_q2.step()
                    for param_group in optimizer_actor.param_groups:
                        print(f"Actor learning rate: {param_group['lr']}")

                    for param_group in optimizer_q1.param_groups:
                        print(f"Critic learning rate: {param_group['lr']}")
                        # Logging (optional)
                    print("ent_coef: ", ent_coef)
                    print("log_entropy: ", log_alpha)
                    print("policy_loss: ", policy_loss)
                    print("q_value_loss: ", q_value_loss)
                    print("ent_coef_loss: ", ent_coef_loss)
                    
                    # Update target networks
                    if global_step % self.hypp.update_param_frequency ==0:
                        for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                            target_param.data.copy_(self.hypp.tau * param.data + (1 - self.hypp.tau) * target_param.data)
                    ########################################################
                    #self.train_actor(self.policy, self.model_rb,self.env_rb,ent_coef, log_alpha,log_entropy,update_param)
                    scheduler.step()
                    scheduler_q1.step()  
                    scheduler_q2.step()  
                print("model_loss, ", model_loss)

        # one last evaluation stage
        if self.exp.eval_agent:
            tracked_return, tracked_episode_len = hf.evaluate_agent(env_eval,policy, self.exp.eval_count, self.exp.seed, greedy_actor = True)
            tracked_returns_over_training.append(tracked_return)
            tracked_episode_len_over_training.append(tracked_episode_len)
            tracked_episode_count.append([episode_step, global_step])

            # if there has been improvement of the model - save model, create video, log video to wandb
            if np.mean(tracked_return) > eval_max_return:
                eval_max_return = np.mean(tracked_return)
                # call helper function save_and_log_agent to save model, create video, log video to wandb
                hf.save_and_log_agent(self.exp, self.policy, episode_step,
                                    greedy=True, print_path=False)

            hf.save_tracked_values(tracked_returns_over_training, tracked_episode_len_over_training, tracked_episode_count, self.exp.eval_count, self.exp.run_name)


        env.close()
        self.writer.close()
        steps.close()
        if wandb.run is not None:
            wandb.finish(quiet=True)
            wandb.init(mode= 'disabled')

        hf.save_train_config_to_yaml(self.exp, self.hypp)

        if self.hypp.plot_training:
            eval_params = edict()  # eval_params - evaluation settings for trained agent

            eval_params.run_name00 = self.exp.run_name
            eval_params.exp_type00 = self.exp.exp_type

            # eval_params.run_name01 = "CartPole-v1__PPO__1__230302_224624"
            # eval_params.exp_type01 = None

            # eval_params.run_name02 = "CartPole-v1__PPO__1__230302_221245"
            # eval_params.exp_type02 = None

            agent_labels = []

            episode_axis_limit = None

            hf.plotter_agents_training_stats(eval_params, agent_labels, episode_axis_limit, plot_returns=True, plot_episode_len=True)
        if self.hypp.display_evaluation:
            print("Agent")
            agent_name =self. exp.run_name
            agent_exp_type = self.exp.exp_type  # both are needed to identify the agent location
            

            exp_folder = "" if agent_exp_type is None else agent_exp_type
            filepath, _ = hf.create_folder_relative(f"{exp_folder}/{agent_name}/videos")

            hf.record_video(self.exp.env_id, agent_name, f"{filepath}/best.mp4", exp_type=agent_exp_type, greedy=True)

            while True:
                #This is to check whether to break the first loop
                isclosed=0
                cap = cv2.VideoCapture(f"{filepath}/best.mp4")
                while (True):

                    ret, frame = cap.read()
                    # It should only show the frame when the ret is true
                    if ret == True:
                        time.sleep(0.3)
                        cv2.imshow('frame',frame)
                        #time.sleep(1)
                        if cv2.waitKey(1) == 27:
                            # When esc is pressed isclosed is 1
                            isclosed=1
                            break
                    else:
                        break
                
                # To break the loop if it is closed manually
                if isclosed:
                    break
                cap.release()
                cv2.destroyAllWindows()
            
