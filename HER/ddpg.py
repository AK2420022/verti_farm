
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
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=DeprecationWarning)
    from torch.utils.tensorboard import SummaryWriter
import utils.helper_fns as hf
import Actor as Actor
import Critic as Critic
from her_simple import HerBuffer
import gymnasium as gym
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
from stable_baselines3.common.buffers import ReplayBuffer
from utils.ddpg_helper_fn import lr_lambda,create_folder_relative,save_tracked_values,save_train_config_to_yaml, make_env,setup_logging,get_noisy_action,update_noisy_process,evaluate_agent,save_and_log_agent,record_video
from tqdm import tqdm
import cv2
import time
from stable_baselines3.common.vec_env import VecEnv
device = torch.device("cpu")
from torch.optim import lr_scheduler

class ddpg():
    def __init__(self,exp =edict(),hypp=edict(),noisep=edict()):

        # Init tensorboard logging and wandb logging
        self.writer = setup_logging(f"{exp.env_id}", exp, hypp)
        #initialize buffer
        env = gym.vector.SyncVectorEnv([make_env(exp.env_id ,exp.seed + i, exp.tasks,exp.terminate_on_tasks_completed,max_episode_steps=exp.max_episode_steps ) for i in range(1)])
        
        if exp.use_HER :
            self.rb = HerBuffer(
                env,
                hypp.buffer_size,
                1,
                env.observation_space,
                env.single_action_space,
                device,
                hypp.batch_size,
                exp.goal_selection_strategy 
            )
        else:
            self.rb = ReplayBuffer(
            hypp.buffer_size,
            env.single_observation_space,
            env.single_action_space,
            device,
        )
        
        env.close()
        self.exp = exp
        self.hypp = hypp
        self.noisep = noisep
        self.total_rewards = 0
    def train(self):
        # create two vectorized envs: one to fill the rollout buffer with trajectories and
        # another one to evaluate the agent performance at different stages of training
        # Note: vectorized environments reset automatically once the episode is finished
        env = gym.vector.SyncVectorEnv([make_env(self.exp.env_id, self.exp.seed,self.exp.tasks,self.exp.terminate_on_tasks_completed ,max_episode_steps=self.exp.max_episode_steps)])
        env_eval = gym.vector.SyncVectorEnv([make_env(self.exp.env_id, self.exp.seed + i,self.exp.tasks,self.exp.terminate_on_tasks_completed,max_episode_steps=self.exp.max_episode_steps ) for i in range(1)])

        # init list to track agent's performance throughout training
        tracked_returns_over_training = []
        tracked_episode_len_over_training = []
        tracked_episode_count = []
        last_evaluated_episode = None  # stores the episode_step of when the agent's performance was last evaluated
        eval_max_return = -float('inf')

        # Init observation to start learning
        start_time = time.time()
        obs = env.reset()

        pbar = tqdm(range(1, self.hypp.total_timesteps + 1))
        # ------------------------- END RUN INIT --------------------------- #

        # Create Actor class Instance and network optimizer
        actor = Actor.Actor(env).to(device)
        actor_target = copy.deepcopy(actor)

        optimizer_actor = optim.AdamW(actor.parameters(), lr=self.hypp.learning_rate,weight_decay=self.hypp.l2_reg)
        # Create Critic class Instance and network optimizer
        critic =Critic.Critic(env).to(device)
        critic_target = copy.deepcopy(critic)

        optimizer_critic = optim.Adam(critic.parameters(), lr=self.hypp.learning_rate_critic,weight_decay=self.hypp.l2_reg)
        #scheduler = lr_scheduler.StepLR(optimizer_actor,step_size=self.hypp.e_steps,gamma=0.1)

        # Initialize Ornstein-Uhlenbeck Process
        noise_process = np.zeros(env.action_space.shape)
        noise_scale = 0.1
        #training
        global_step = 0
        episode_step = 0
        gradient_step = 0
        #training loop
        current_obs = obs[0]["observation"]
        for update in pbar:
            #print("obs: ",type(obs))
            #print("goal: ",goal)
            noisy_action = get_noisy_action(actor,current_obs,env,noise_process,noise_scale,self.noisep)
            noise_process = update_noisy_process(noise_process,env,self.noisep)
            # apply action to environment
            next_obs, reward, done, truncated,infos = env.step(noisy_action.numpy())
            dones = done or truncated
            #print("next_obs m1: ",next_obs["observation"][0][51])
            #print("next_obs m2: ",next_obs["observation"][0][30])
            global_step += 1
            self.total_rewards += reward
            print("total reward: ",self.total_rewards)
            #print("noisy action: ",noisy_action)
            # log episode return and length to tensorboard as well as current epsilon
            
            for info,value in infos.items():
                if info == "final_info" :
                    
                    for key,values in value[0].items():
                        if key == "episode":
                            #print(info)
                            #print(values)
                            
                            episode_step += 1
                            pbar.set_description(f"global_step: {global_step}, episodic_return={values['r']}")
                            self.writer.add_scalar("rollout/episodic_return", values["r"], global_step)
                            self.writer.add_scalar("rollout/episodic_length", values["l"], global_step)
                            self.writer.add_scalar("Charts/episode_step", episode_step, global_step)
                            self.writer.add_scalar("Charts/gradient_step", gradient_step, global_step)
            # evaluation of the agent
            if self.exp.eval_agent and (global_step % self.exp.eval_frequency == 0) and last_evaluated_episode != episode_step:
                last_evaluated_episode = episode_step
                tracked_return, tracked_episode_len = evaluate_agent(env_eval, actor, self.exp.eval_count,
                                                                       self.exp.seed, self.exp.tasks,greedy_actor=True)
                tracked_returns_over_training.append(tracked_return)
                tracked_episode_len_over_training.append(tracked_episode_len)
                tracked_episode_count.append([episode_step, global_step])

                # if there has been improvement of the model - save model, create video, log video to wandb
                if np.mean(tracked_return) > eval_max_return:
                    eval_max_return = np.mean(tracked_return)
                    print("eval_max_return")
                    # call helper function save_and_log_agent to save model, create video, log video to wandb
                    save_and_log_agent(self.exp, actor, episode_step,
                                        greedy=True, print_path=False)
            # handling the terminal observation (vectorized env would skip terminal state)
            real_next_obs = next_obs.copy()
            
            for idx, d in enumerate(done):
                if d:
                    self.total_rewards = 0
                    #real_next_obs[idx] = infos[idx]["terminal_observation"]
            
            # add data to replay buffer
            self.rb.add(obs, real_next_obs, noisy_action, reward, dones, infos)
            # update obs
            current_obs = next_obs["observation"].squeeze()
            # training of the agent
            if global_step > self.hypp.start_learning:
                if global_step % self.hypp.train_frequency == 0:
                    data = self.rb.sample(self.hypp.batch_size)

                    q_target_actor =get_noisy_action(actor_target,data.next_observations["observation"].squeeze(),env,noise_process,noise_scale,self.noisep) 
                    q_target_critic = critic_target(data.next_observations["observation"].squeeze(),q_target_actor)
                    #print("qtc: ", q_target_critic , " reward: ", data.rewards)
                    q_target_critic, _ = torch.min(q_target_critic, dim=1, keepdim=True)
                    c_target = self.hypp.reward_scaling * data.rewards + self.hypp.gamma * q_target_critic * (1  - data.dones)
                    # Compute critic loss
                    #print("q_critic: ",q_target_actor[0])

                    #print("ctarget: ",q_target_critic[0])
                    q_critic = critic(data.observations["observation"].squeeze(),data.actions)

                    critic_loss = sum(F.mse_loss(c_q,c_target.to(torch.float32)) for c_q in q_critic.to(torch.float32))         
                    print("critic_loss: ",critic_loss)
                    #optmize critic
                    optimizer_critic.zero_grad()
                    critic_loss.backward()
                    optimizer_critic.step()
                    
                    if global_step % self.hypp.policy_delay == 0:

                        #compute actor loss
                        q_value_actor = actor(data.observations["observation"].squeeze())
                        q_value_critic = critic(data.observations["observation"].squeeze(),q_value_actor)
                        #print("q_value_critic: ",q_value_critic)
                        actor_loss =  - q_value_critic.mean()
                        print("actor loss: ",actor_loss)
                        #optmize the actor 
                        optimizer_actor.zero_grad()
                        actor_loss.backward()
                        optimizer_actor.step()

                        # log critic_loss and q_values to tensorboard
                        if global_step % 100 == 0:
                            self.writer.add_scalar("train/critic_loss", critic_loss, global_step)
                            self.writer.add_scalar("train/actor_loss", actor_loss, global_step)
                            self.writer.add_scalar("others/SPS", int(global_step / (time.time() - start_time)), global_step)
                            self.writer.add_scalar("Charts/episode_step", episode_step, global_step)
                            self.writer.add_scalar("Charts/gradient_step", gradient_step, global_step)
                        # Update the frozen target models
                        for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                            target_param.data.copy_(self.hypp.tau * param.data + (1 - self.hypp.tau) * target_param.data)

                        for param, target_param in zip(actor.parameters(), actor_target.parameters()):
                            target_param.data.copy_(self.hypp.tau* param.data + (1 - self.hypp.tau) * target_param.data)
                    #scheduler.step()
        # one last evaluation stage
        if self.exp.eval_agent:
            tracked_return, tracked_episode_len = evaluate_agent(env_eval, actor, self.exp.eval_count, self.exp.seed, self.exp.tasks,greedy_actor = True)
            tracked_returns_over_training.append(tracked_return)
            tracked_episode_len_over_training.append(tracked_episode_len)
            tracked_episode_count.append([episode_step, global_step])

            # if there has been improvement of the model - save model, create video, log video to wandb
            if np.mean(tracked_return) >= eval_max_return:
                eval_max_return = np.mean(tracked_return)
                # call helper function save_and_log_agent to save model, create video, log video to wandb
                save_and_log_agent(self.exp, actor, episode_step,
                                    greedy=True, print_path=False)

            save_tracked_values(tracked_returns_over_training, tracked_episode_len_over_training, tracked_episode_count, self.exp.eval_count, self.exp.run_name)
                        
        env.close()
        self.writer.close()
        pbar.close()
        if wandb.run is not None:
            wandb.finish(quiet=True)
            wandb.init(mode= 'disabled')

        save_train_config_to_yaml(self.exp, self.hypp)    

        if self.hypp.display_evaluation:
            agent_name = self.exp.run_name
            agent_exp_type = self.exp.exp_type  # both are needed to identify the agent location
            

            exp_folder = "" if agent_exp_type is None else agent_exp_type
            filepath, _ = create_folder_relative(f"{exp_folder}/{agent_name}/videos")

            record_video(self.exp.env_id, agent_name, f"{filepath}/best.mp4", exp_type=agent_exp_type, greedy=True)


            

            cap = cv2.VideoCapture(f"{filepath}/best.mp4")
            while True:
                #This is to check whether to break the first loop
                isclosed=0
                cap = cv2.VideoCapture(f"{filepath}/best.mp4")
                while (True):

                    ret, frame = cap.read()
                    # It should only show the frame when the ret is true
                    if ret == True:
                        
                        cv2.imshow('frame',frame)
                        time.sleep(0.05)
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