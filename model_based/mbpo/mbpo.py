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
import torch.distributions as dist
################################################################
from omegaconf import OmegaConf
#testing with mbrl
"""
import mbrl.examples
import mbrl.examples.conf
import mbrl.examples.conf.overrides
import mbrl.constants
import mbrl.models
import mbrl.planning
import mbrl.third_party.pytorch_sac_pranz24 as pytorch_sac_pranz24
import mbrl.types
import mbrl.util
import mbrl.util.common
import mbrl.util.math
"""
################################################################
#import HalfCheetahEnv as cheetah_env
warnings.filterwarnings("ignore", category=DeprecationWarning)

os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['WANDB_NOTEBOOK_NAME'] = 'dqn.py'

plt.rcParams['figure.dpi'] = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        
        self.run_name = f"{exp.env_id}__{exp.exp_name}__{exp.seed}__{datetime.now().strftime('%y%m%d_%H%M%S')}"
        if exp is not None:
            self.exp = exp
        if hypp is not None:
            self.hypp = hypp
        # Init tensorboard logging and wandb logging
        self.writer = hf.setup_logging(f"{exp.env_id}", exp, hypp)
        self._rollout_length = self.hypp.num_rollouts
        self._rollout_schedule = self.hypp.rollout_schedule
        self.angle_threshold = 0.2
        self.action_penalty = 0.5
        self.directional_reward = torch.full(fill_value=1,size=(self.hypp.num_rollouts, 1,1))
        self.prev_pole_angle = np.zeros(shape=(self.hypp.num_rollouts, 1,1))
        self.prev_action = np.zeros(shape=(self.hypp.num_rollouts, 1,1))
        ################################ mbrl

        #self.cheetah_env =
        #  cheetah_env.HalfCheetahEnv()
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
    def generate_data(self,model_env,env_rb,samples = 500):
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
        #obs = env.reset() 
        #obs = obs[0]
        print("Rollout Length: ", self._rollout_length)
        for _ in range(self._rollout_length):
            #sample st from Denv
            with torch.no_grad():
                action, log_prob = policy.get_action(states)
                next_obs, reward = ensemble.sample_predictions(states,action)
            next_obs = next_obs
            #reward function (optional reward?)
            _,done = self.compute_reward(env,states,next_obs,action,reward)
            """
            ##################################3
            
            action,log_prob = policy.get_action(states)
            
            for i in range(action.shape[0]):
                # apply action to environment
                next_obs, reward, done,truncated, infos = env.step(action[i].squeeze(1).cpu().detach().numpy())
                done = np.logical_or( done ,  truncated)
                model_buffer.add(states[i],next_obs,action[i].cpu().detach().numpy(),reward,done,{})
                #not_done_mask = ~done
            #self.insert_batch(model_buffer,states[not_done_mask],next_obs[not_done_mask],action[not_done_mask],reward[not_done_mask],done[not_done_mask],{})
            """
            ##################################
            #print("pole angle: " , pole_angle)
        
            truncated = False if episode_step < 1000 else True
            done = np.logical_or( done ,  truncated)
            not_done_mask = ~done

            self.insert_batch(model_buffer,states[not_done_mask].squeeze(1),next_obs[not_done_mask],action[not_done_mask],reward[not_done_mask],done[not_done_mask],{})
            
            states = next_obs
            accum_dones =(accum_dones | done).to(torch.bool)
            
            episode_step +=1
            # Reset states if necessary
            # Break if all rollouts are done
            if done.all():
                break

    def compute_reward(self,env,state,next_state,action,reward,info = None):
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
        if self.exp.env_id == "HalfCheetah-v4":

            x_vel_curr = state.squeeze(1)[:,0]
            x_vel_next = next_state[:,0]
            

            #rewards = env.call("_get_rew",x_vel,action)
            dt =  env.envs[0].model.opt.timestep
            x_vel = ( x_vel_next - x_vel_curr ) / dt
            
            #print("reward: ", (x_vel_next - x_vel_curr ))
            _forward_reward_weight = 1.0
            forward_reward = _forward_reward_weight * x_vel 
            
            #reward = rewards[0][0] 
            _ctrl_cost_weight = 0.1 
            control_cost = _ctrl_cost_weight * torch.sum(torch.square(action), dim=-1)
            stability_reward_weight = 0.05
            stability_reward = -stability_reward_weight * torch.sum(torch.abs(action[:,1:] - action[:, :-1]), dim=-1)
            
            reward = forward_reward - control_cost + stability_reward

            # Reward scaling and clipping to prevent overly large or small values
            reward = torch.clamp(reward, min=-10, max=10)
            reward = reward.view(forward_reward.size(dim=0),1)

            # Optional: Add bonus for consistent forward movement
            bonus_threshold = 1.0
            if x_vel.mean() > bonus_threshold:
                reward += 5.0
            #if rewards[0][1]["reward_forward"][0] < 0 :
            #    reward = rewards[0][1]["reward_forward"][0] * 0.8 - rewards[0][1]["reward_ctrl"] * 0.5
            done = torch.zeros(size=(state.size(dim=0),), dtype=torch.bool)
            
            return reward, done
        elif self.exp.env_id == "InvertedPendulum-v4":
            pole_angle = state[:,:,1]
            #print("pole angle, ",pole_angle)
            #absolute_change = old_angle - pole_angle
            # Penalize based on the pole angle
            #print("pole angle, ",pole_angle)
            angle_penalty = pole_angle ** 2
            angle_penalty = angle_penalty[:,:,np.newaxis]
            pole_angle = pole_angle[:,:,np.newaxis].cpu().detach().numpy()
            # Penalize based on the action magnitude
            action_penalty = self.action_penalty * (action ** 2)
            #print("angle_penalty, ",angle_penalty)
            # Initialize the reward
            reward -=  (angle_penalty + action_penalty)
            #print("reward, ",reward)
            
            result = ""
            angle_threshold = self.angle_threshold
            prev_angle = pole_angle[0]
            for i,(pole_ang,act) in enumerate(zip(pole_angle,action)): 
                #print("pole angle, ",pole_ang)
                #print("pole action, ",act)
                if abs(pole_ang) > angle_threshold:
                    result = "doomed"
                    print(result)
                    reward[i] -=100
                       
                else:    
                    # Check if the pole angle increased and the action is in the same direction as the previous action
                    if abs(pole_ang) > abs(prev_angle) and np.sign(act) == np.sign(prev_act):
                        # Penalize if the pole angle increased and action is in the same direction
                        reward[i] -= 0.1#reward[i] -= self.directional_reward[i] 
                        result = "penalize"
                    elif abs(pole_ang) > abs(prev_angle) and np.sign(act) != np.sign(prev_act):
                        # Reward if the pole angle increased and action is in the opposite direction
                        reward[i] += 1#reward[i] += self.directional_reward[i]
                        result = "reward"
                prev_angle = pole_ang
                #print("pole angle_prev_new, ",prev_angle)
                prev_act = act
                
                #print(result)
                i+=1

            
            # Update the previous angle and action
            self.prev_pole_angle = pole_angle
            self.prev_action = action
            #reward = 1.0 if np.abs(pole_angle) < 0.2 else -1
            done = (abs(pole_angle) > self.angle_threshold)
            return reward.cpu().detach().numpy(), done
        
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
            model_buffer.add(obs[i].cpu().detach().numpy(),next_obs[i].cpu().detach().numpy(),action[i].cpu().detach().numpy(),reward[i].cpu().detach().numpy(),done[i],{})
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
        return 1 - epoch / (self.hypp.total_timesteps)

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
        # Initialize environments
        env = gym.vector.SyncVectorEnv([hf.make_env(self.exp.env_id, self.exp.seed)])
        env_eval = gym.vector.SyncVectorEnv([hf.make_env(self.exp.env_id, self.exp.seed + i) for i in range(1)])
        
        # Check and adjust learning start
        if self.hypp.start_learning >= self.hypp.total_timesteps:
            self.hypp.start_learning = self.hypp.total_timesteps / 2 
        
        # Initialize entropy coefficient
        log_entropy = self.hypp.log_entropy if self.hypp.log_entropy is not None else -np.array(env.action_space.shape).prod()
        log_alpha = torch.tensor(0.0, requires_grad=True)
        ent_coef = log_alpha.exp()
        alpha_optimizer = optim.Adam(params=[log_alpha], lr=self.hypp.learning_rate)
        scheduler_alpha = torch.optim.lr_scheduler.LambdaLR(alpha_optimizer, lr_lambda=self.lambda_schedule)

        # Initialize model
        model_env = Ensemble(env, self.hypp.learning_rate_model, num_ensembles=self.hypp.num_ensembles, num_steps=self.hypp.total_timesteps).to(device)

        # Initialize policy and optimizers
        policy = Actor(env, self.hypp.hidden_dim, self.hypp.hidden_layers_actor).to(device)
        policy.init_weights()
        optimizer_actor = optim.Adam(policy.parameters(), lr=self.hypp.learning_rate_actor)
        scheduler_actor = torch.optim.lr_scheduler.LambdaLR(optimizer_actor, lr_lambda=self.lambda_schedule)

        # Initialize critic
        critic = Critic(env, self.hypp.hidden_dim, self.hypp.hidden_layers_critic).to(device)
        critic_target = copy.deepcopy(critic).to(device)
        optimizer_q1 = optim.Adam(critic.q1.parameters(), lr=self.hypp.learning_rate_critic)
        optimizer_q2 = optim.Adam(critic.q2.parameters(), lr=self.hypp.learning_rate_critic)
        scheduler_q1 = torch.optim.lr_scheduler.LambdaLR(optimizer_q1, lr_lambda=self.lambda_schedule)
        scheduler_q2 = torch.optim.lr_scheduler.LambdaLR(optimizer_q2, lr_lambda=self.lambda_schedule)

        # Track metrics
        tracked_returns_over_training = []
        tracked_episode_len_over_training = []
        tracked_episode_count = []
        last_evaluated_episode = None
        eval_max_return = -float('inf')
        model_loss = 0
        start_time = time.time()
        
        # Initialize observation
        obs = env.reset()[0]
        global_step = 0
        episode_step = 0
        gradient_step = 0
        
        # Pre-fill buffer
        self.generate_data(env, self.env_rb, samples=5000)
        
        # Training loop
        for epoch in range(self.hypp.num_epochs):
            episode_reward = 0
            episode_len = 0
            self._set_rollout_length(epoch)  # Adjust rollout length based on the epoch
            
            for timestep in tqdm(range(self.hypp.epoch_length)):
                # Take action and interact with environment
                action, log_prob = policy.get_action(torch.tensor(obs).to(torch.float32))
                next_obs, reward, done, truncated, infos = env.step(action.cpu().detach().numpy())
                done = done or truncated or (global_step % 1000 == 0)
                
                # Update buffers
                self.env_rb.add(obs, next_obs, action.cpu().detach().numpy(), reward, done, infos)
                obs = next_obs if not done else env.reset()[0]
                global_step += 1
                episode_reward += reward
                episode_len += 1
                print("Current episode: ", episode_step)
                # Log episode metrics
                for info, values in infos.items():

                    if "final_info" == info:
                        print("comes here values: " ,values)
                        print("comes here info: " ,info)
                        episode_step += 1
                        self.writer.add_scalar("rollout/episodic_return", values[0]['episode']["r"][0], global_step)
                        self.writer.add_scalar("rollout/episodic_length", values[0]['episode']["l"][0], global_step)
                        self.writer.add_scalar("Charts/episode_step", episode_step, global_step)
                        self.writer.add_scalar("Charts/gradient_step", gradient_step, global_step)
                        break
                
                # Evaluate agent periodically
                if self.exp.eval_agent and global_step % self.exp.eval_frequency == 0 and last_evaluated_episode != episode_step and global_step >= self.hypp.start_learning:
                    last_evaluated_episode = episode_step
                    tracked_return, tracked_episode_len = hf.evaluate_agent(env_eval, policy, self.exp.eval_count, self.exp.seed, greedy_actor=True)
                    tracked_returns_over_training.append(tracked_return)
                    tracked_episode_len_over_training.append(tracked_episode_len)
                    tracked_episode_count.append([episode_step, global_step])
                    if np.mean(tracked_return) > eval_max_return:
                        eval_max_return = np.mean(tracked_return)
                        hf.save_and_log_agent(self.exp, policy, episode_step, greedy=True, print_path=False)

                # Model rollouts and updates
                if global_step > self.hypp.rollout_start and global_step % self.hypp.train_frequency == 0:
                    self.do_rollouts(env, policy, model_env, self.env_rb, self.model_rb, self.hypp.num_rollouts)

                # Update model
                if global_step > self.hypp.start_learning and global_step % self.hypp.model_train_frequency == 0:
                    model_loss, trained_epochs = model_env.train_step(self.env_rb.sample(self.hypp.batch_size))
                
                # Update actor and critic
                if global_step > self.hypp.start_learning and global_step % self.hypp.train_frequency == 0:
                    real_data_size = int(self.hypp.batch_size * self.hypp.real_img_ratio)
                    img_data_size = self.hypp.batch_size - real_data_size
                    real_data = self.env_rb.sample(real_data_size)
                    img_data = self.model_rb.sample(img_data_size)
                    
                    data = ReplayBufferSamples(
                        observations=torch.cat((real_data.observations, img_data.observations), dim=0),
                        actions=torch.cat((real_data.actions, img_data.actions), dim=0),
                        next_observations=torch.cat((real_data.next_observations, img_data.next_observations), dim=0),
                        dones=torch.cat((real_data.dones, img_data.dones), dim=0),
                        rewards=torch.cat((real_data.rewards, img_data.rewards), dim=0)
                    )
                    
                    rewards = data.rewards
                    with torch.no_grad():
                        new_action, new_log_prob = policy.get_action(data.next_observations.to(torch.float32).squeeze(1), deterministic=False)
                        new_q_value1, new_q_value2 = critic_target(data.next_observations.squeeze(1), new_action)
                        new_q_value_min = torch.min(new_q_value1, new_q_value2)
                        new_q_value_target = new_q_value_min.detach() - ent_coef * new_log_prob.detach()
                    reward_scale = 0.5
                    target_q_value = self.scale(rewards) + self.hypp.gamma * new_q_value_target * (1 - data.dones)
                    target_q_value = target_q_value.detach()
                    q1, q2 = critic(data.observations.squeeze(1), data.actions)
                    q_value_loss =  (F.mse_loss(q1, target_q_value) + F.mse_loss(q2, target_q_value))
                    
                    optimizer_q1.zero_grad()
                    optimizer_q2.zero_grad()
                    q_value_loss.backward()
                    clip_grad_norm_(critic.parameters(), 1)
                    optimizer_q1.step()
                    optimizer_q2.step()
                    
                    # Update target networks
                    if global_step % self.hypp.update_param_frequency == 0:
                        for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                            target_param.data.copy_(self.hypp.tau * param.data + (1 - self.hypp.tau) * target_param.data)
                    
                    # Update actor
                    if global_step % self.hypp.train_frequency == 0:
                        actions, log_prob = policy.get_action(data.observations.squeeze(1).to(torch.float32), deterministic=False)
                        policy_prior_log_probs = 0
                        if self.hypp.action_prior == 'normal':
                            policy_prior = dist.MultivariateNormal(
                                loc=torch.zeros(env.action_space.shape),
                                covariance_matrix=torch.eye(env.action_space.shape[0])
                            )
                            policy_prior_log_probs = policy_prior.log_prob(actions)
                        elif self.hypp.action_prior == 'uniform':
                            policy_prior_log_probs = torch.tensor(0.0)
                        
                        q1_actor, q2_actor = critic(data.observations, actions)
                        min_q_actor = torch.min(q1_actor, q2_actor)
                        policy_loss = torch.mean(ent_coef * log_prob - min_q_actor - policy_prior_log_probs)
                        
                        optimizer_actor.zero_grad()
                        policy_loss.backward(retain_graph=True)
                        optimizer_actor.step()
                        
                        # Update entropy coefficient
                        
                        ent_coef_loss = torch.mean((ent_coef * (-log_entropy - log_prob).detach()))
                        alpha_optimizer.zero_grad()
                        ent_coef_loss.backward()
                        grad_norm = log_alpha.grad.abs()
                        if grad_norm > 0.5:
                            log_alpha.grad *= 1 / grad_norm
                   
                        alpha_optimizer.step()
                        #scheduler_alpha.step()
                        ent_coef = torch.exp(log_alpha)
                        print("model_loss, ", model_loss)
                        print("ent_coef: ", ent_coef)
                        print("log_entropy: ", log_alpha)
                        print("policy_loss: ", policy_loss)
                        print("q_value_loss: ", q_value_loss)
                        print("ent_coef_loss: ", ent_coef_loss)
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
                hf.save_and_log_agent(self.exp, policy, episode_step,
                                    greedy=True, print_path=False)

            hf.save_tracked_values(tracked_returns_over_training, tracked_episode_len_over_training, tracked_episode_count, self.exp.eval_count, self.exp.run_name)


        env.close()
        self.writer.close()
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

            
                #This is to check whether to break the first loop
            cap = cv2.VideoCapture(f"{filepath}/best.mp4")

            while cap.isOpened():
                ret, frame = cap.read()  # Read the next frame from the video

                if ret:
                    cv2.imshow('Video', frame)  # Display the frame in a window

                    # Wait for 25ms before moving to the next frame
                    # This will display the video at approximately its original frame rate
                    if cv2.waitKey(25) & 0xFF == 27:  # Check if 'Esc' key is pressed
                        break
                else:
                    break  
                
            

            cap.release()
            cv2.destroyAllWindows()
            return True        
        
    def lambda_schedule(self,epoch):
        return max(0.99 ** epoch, 0.1)
