
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
import gymnasium as gym
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
from stable_baselines3.common.buffers import ReplayBuffer
from ddpg_helper_fn import get_noisy_action,update_noisy_process,evaluate_agent
from tqdm import tqdm
import cv2
import time
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['WANDB_NOTEBOOK_NAME'] = 'ddpg.ipynb'

warnings.filterwarnings("ignore", category=DeprecationWarning)

plt.rcParams['figure.dpi'] = 100
device = torch.device("cpu")

exp = edict()

exp.exp_name = 'DDPG'  # algorithm name, in this case it should be 'DQN'
exp.env_id = 'InvertedPendulum-v2'  # name of the gym environment to be used in this experiment. Eg: Acrobot-v1, CartPole-v1, MountainCar-v0
exp.device = device.type  # save the device type used to load tensors and perform tensor operations

set_random_seed = True  # set random seed for reproducibility of python, numpy and torch
exp.seed = 2

# name of the project in Weights & Biases (wandb) to which logs are patched. (only if wandb logging is enabled)
# if the project does not exist in wandb, it will be created automatically
wandb_prj_name = f"RLLBC_{exp.env_id}"

# name prefix of output files generated by the notebook
exp.run_name = f"{exp.env_id}__{exp.exp_name}__{exp.seed}__{datetime.now().strftime('%y%m%d_%H%M%S')}"

if set_random_seed:
    random.seed(exp.seed)
    np.random.seed(exp.seed)
    torch.manual_seed(exp.seed)
    torch.backends.cudnn.deterministic = set_random_seed

# initialize the parameters
hypp = edict()

# flags for logging purposes
exp.enable_wandb_logging = True
exp.capture_video = True

# flags to generate agent's average performance during training
exp.eval_agent = True  # disable to speed up training
exp.eval_count = 10
exp.eval_frequency = 1000
exp.use_HER = True 
hypp.train_frequency = 1
# putting the run into the designated log folder for structuring
exp.exp_type = None  # directory the run is saved to. Should be None or a string value

# agent training specific parameters and hyperparameters
hypp.total_timesteps = 100000  # the training duration in number of time steps
hypp.learning_rate = 2.5e-4  # the learning rate for the optimizer
hypp.gamma = 0.99  # decay factor of future rewards
hypp.buffer_size = 50000  # the size of the replay memory buffer
hypp.target_network_frequency = 500  # the frequency of synchronization with target network
hypp.batch_size = 128# number of samples taken from the replay buffer for one step
hypp.start_e = 1  # probability of exploration (epsilon) at timestep 0
hypp.end_e = 0.01  # minimal probability of exploration (epsilon)
hypp.exploration_fraction =0.5 # the fraction of total_timesteps it takes to go from start_e to end_e
hypp.start_learning = 10000  # the timestep the learning 
hypp.tau = 0.001
noisep = edict()
# Define Ornstein-Uhlenbeck Process parameters
noisep.theta = 0.15
noisep.sigma = 0.2
noisep.dt = 1e-2
noisep.mu = 0
hypp.display_evaluation = True #display video evaluation
hypp.plot_training = True # plot training
# replay buffer parameters
#initialize buffer
env = gym.vector.SyncVectorEnv([hf.make_env(exp.env_id, exp.seed + i) for i in range(1)])
rb = ReplayBuffer(
    hypp.buffer_size,
    env.single_observation_space,
    env.single_action_space,
    device,
    handle_timeout_termination = False,
)

env.close()

# reinit run_name
exp.run_name = f"{exp.env_id}__{exp.exp_name}__{exp.seed}__{datetime.now().strftime('%y%m%d_%H%M%S')}"

# Init tensorboard logging and wandb logging
writer = hf.setup_logging(wandb_prj_name, exp, hypp)

# create two vectorized envs: one to fill the rollout buffer with trajectories and
# another one to evaluate the agent performance at different stages of training
# Note: vectorized environments reset automatically once the episode is finished
env = gym.vector.SyncVectorEnv([hf.make_env(exp.env_id, exp.seed)])
env_eval = gym.vector.SyncVectorEnv([hf.make_env(exp.env_id, exp.seed + i) for i in range(1)])

# init list to track agent's performance throughout training
tracked_returns_over_training = []
tracked_episode_len_over_training = []
tracked_episode_count = []
last_evaluated_episode = None  # stores the episode_step of when the agent's performance was last evaluated
eval_max_return = -float('inf')

# Init observation to start learning
start_time = time.time()
obs = env.reset()
obs = obs[0]
pbar = tqdm(range(1, hypp.total_timesteps + 1))

# ------------------------- END RUN INIT --------------------------- #

# Create Actor class Instance and network optimizer
actor = Actor.Actor(env).to(device)
actor_target = copy.deepcopy(actor)

optimizer_actor = optim.Adam(actor.parameters(), lr=hypp.learning_rate)
# Create Critic class Instance and network optimizer
critic =Critic.Critic(env).to(device)
critic_target = copy.deepcopy(critic)

optimizer_critic = optim.Adam(critic.parameters(), weight_decay=1e-2)

# Initialize Ornstein-Uhlenbeck Process
noise_process = np.zeros(env.action_space.shape)
noise_scale = 0.5
#training
global_step = 0
episode_step = 0
gradient_step = 0

#training loop
for update in pbar:
    noisy_action = get_noisy_action(actor,obs,env,noise_process,noise_scale,noisep)
    noise_process = update_noisy_process(noise_process,env,noisep)
    # apply action to environment
    print(noisy_action.numpy())
    next_obs, reward,truncated, done, infos = env.step(noisy_action.numpy())
    done = done or truncated 

    global_step += 1
    if not infos:
        infos["TimeLimit.truncated"] = False
    # log episode return and length to tensorboard as well as current epsilon
    for info,values in infos.items():
        if "final_info" == info:
            print("global step: ", global_step)
            episode_step += 1
            pbar.set_description(f"global_step: {global_step}, episodic_return={values[0]['episode']['r']}")
            writer.add_scalar("rollout/episodic_return", values[0]["episode"]["r"], global_step)
            writer.add_scalar("rollout/episodic_length", values[0]["episode"]["l"], global_step)
            writer.add_scalar("Charts/episode_step", episode_step, global_step)
            writer.add_scalar("Charts/gradient_step", gradient_step, global_step)
            break
    
    
    # evaluation of the agent
    if exp.eval_agent and (episode_step % exp.eval_frequency == 0) and last_evaluated_episode != episode_step:
        last_evaluated_episode = episode_step
        tracked_return, tracked_episode_len = evaluate_agent(env_eval, actor, exp.eval_count,
                                                                exp.seed, greedy_actor=True)
        tracked_returns_over_training.append(tracked_return)
        tracked_episode_len_over_training.append(tracked_episode_len)
        tracked_episode_count.append([episode_step, global_step])

        # if there has been improvement of the model - save model, create video, log video to wandb
        if np.mean(tracked_return) > eval_max_return:
            eval_max_return = np.mean(tracked_return)
            # call helper function save_and_log_agent to save model, create video, log video to wandb
            hf.save_and_log_agent(exp, actor, episode_step,
                                  greedy=True, print_path=False)
    # handling the terminal observation (vectorized env would skip terminal state)
    real_next_obs = next_obs.copy()
    """
    for idx, d in enumerate(done):
        if d:
            real_next_obs[idx] = infos[idx]["terminal_observation"]
    """
    # add data to replay buffer
    #TODO handle timeout in a different way. up to stable baseline
    rb.add(obs, real_next_obs, noisy_action, reward, done, infos)
    # update obs
    obs = next_obs
    # training of the agent
    if global_step > hypp.start_learning:
        if global_step % hypp.train_frequency == 0:
            data = rb.sample(hypp.batch_size)
            q_target_actor = get_noisy_action(actor_target,data.next_observations,env,noise_process,noise_scale,noisep) 
            q_target_critic = critic_target(data.next_observations,q_target_actor)
            q_target_critic, _ = torch.min(q_target_critic, dim=1, keepdim=True)
            q_critic = critic(data.observations,data.actions)
            
            c_target = data.rewards + hypp.gamma * q_target_critic * (1  - data.dones)
    
            # Compute critic loss
            critic_loss = sum(F.mse_loss(c_q,c_target.to(torch.float32)) for c_q in q_critic.to(torch.float32))         
            #optmize critic
            optimizer_critic.zero_grad()
            critic_loss.backward()
            optimizer_critic.step()
            

            #compute actor loss
            q_value_actor = actor(data.observations)
            q_value_critic = critic(data.observations,q_value_actor)
            actor_loss =  - q_value_critic.mean()
            
            #optmize the actor 
            optimizer_actor.zero_grad()
            actor_loss.backward()
            optimizer_actor.step()

            # log critic_loss and q_values to tensorboard
            if global_step % 100 == 0:
                writer.add_scalar("train/critic_loss", critic_loss, global_step)
                writer.add_scalar("train/actor_loss", actor_loss, global_step)
                writer.add_scalar("others/SPS", int(global_step / (time.time() - start_time)), global_step)
                writer.add_scalar("Charts/episode_step", episode_step, global_step)
                writer.add_scalar("Charts/gradient_step", gradient_step, global_step)

            # Update the frozen target models
            for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                target_param.data.copy_(hypp.tau * param.data + (1 - hypp.tau) * target_param.data)

            for param, target_param in zip(actor.parameters(), actor_target.parameters()):
                target_param.data.copy_(hypp.tau* param.data + (1 - hypp.tau) * target_param.data)

# one last evaluation stage
if exp.eval_agent:
    tracked_return, tracked_episode_len = hf.evaluate_agent(env_eval, actor, exp.eval_count, exp.seed, greedy_actor = True)
    tracked_returns_over_training.append(tracked_return)
    tracked_episode_len_over_training.append(tracked_episode_len)
    tracked_episode_count.append([episode_step, global_step])

    # if there has been improvement of the model - save model, create video, log video to wandb
    if np.mean(tracked_return) > eval_max_return:
        eval_max_return = np.mean(tracked_return)
        # call helper function save_and_log_agent to save model, create video, log video to wandb
        hf.save_and_log_agent(exp, actor, episode_step,
                              greedy=True, print_path=False)

    hf.save_tracked_values(tracked_returns_over_training, tracked_episode_len_over_training, tracked_episode_count, exp.eval_count, exp.run_name)
                
env.close()
writer.close()
pbar.close()
if wandb.run is not None:
    wandb.finish(quiet=True)
    wandb.init(mode= 'disabled')

hf.save_train_config_to_yaml(exp, hypp)    

if hypp.display_evaluation:
    print("Agent")
    agent_name = exp.run_name
    agent_exp_type = exp.exp_type  # both are needed to identify the agent location
    

    exp_folder = "" if agent_exp_type is None else agent_exp_type
    filepath, _ = hf.create_folder_relative(f"{exp_folder}/{agent_name}/videos")

    hf.record_video(exp.env_id, agent_name, f"{filepath}/best.mp4", exp_type=agent_exp_type, greedy=True)


    
    while True:
        #This is to check whether to break the first loop
        isclosed=0
        cap = cv2.VideoCapture(f"{filepath}/best.mp4")
        while (True):

            ret, frame = cap.read()
            # It should only show the frame when the ret is true
            if ret == True:

                cv2.imshow('frame',frame)
                if cv2.waitKey(1) == 27:
                    # When esc is pressed isclosed is 1
                    isclosed=1
                    break
            else:
                break
        time.sleep(3)
        # To break the loop if it is closed manually
        if isclosed:
            break
        cap.release()
        cv2.destroyAllWindows()