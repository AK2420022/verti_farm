import configparser
import subprocess
import os
import random
import warnings
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from easydict import EasyDict as edict
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=DeprecationWarning)
    from torch.utils.tensorboard import SummaryWriter
import torch
from mbpo import mbpo
import carb
class train_mbpo(object):
    """_summary_
    Class to train the MBPO (Model-Based Policy Optimization) with PNN Ensemble algorithm,
    with the Vertifarm Environment.

    Attributes:
        exp (EasyDict): The experiment configuration, including environment and experiment parameters.
        hypp (EasyDict): Hyperparameters for training.

    Methods:
        __init__(): Initializes the training object and prepares the experiment and hyperparameters.
        update_parameters(params): Updates experiment and hyperparameters based on the given parameters.
        run_training(): Runs the MBPO training process.
    """

    def __init__(self):
        """_summary_
        Initializes the training object and prepares the experiment and hyperparameters.

        Sets up the `exp` and `hypp` attributes and prepares them for the training process.
        """
        self.exp = edict()
        self.hypp = edict()
        self.env_config = dict()

    def update_parameters(self, params, env_config):
        """_summary_
        Updates experiment and hyperparameters based on the given parameters.

        Args:
            params (dict): A dictionary containing the training parameters to configure the experiment.
        
        Updates the parameters related to environment, logging, training duration, learning rates, and agent settings.
        """
        device = torch.device("cuda")
        
        exp = edict()
        exp.exp_name = 'mbpo'  # algorithm name, in this case it should be 'DQN'
        exp.env_id =  'VertifarmEnv-v0'  # Name of the gym environment.
        exp.device = device.type  # Device used for tensor operations.
        exp.max_episode_steps = 1000
        exp.set_random_seed = True  # Set random seed for reproducibility.
        exp.seed = 2

        # Logging setup
        wandb_prj_name = f"RL_{exp.env_id}"
        exp.run_name = f"{exp.env_id}__{exp.exp_name}__{exp.seed}__{datetime.now().strftime('%y%m%d_%H%M%S')}"

        # Random seed initialization
        if exp.set_random_seed:
            random.seed(exp.seed)
            np.random.seed(exp.seed)
            torch.manual_seed(exp.seed)
            torch.backends.cudnn.deterministic = exp.set_random_seed

        # Initialize hyperparameters
        hypp = edict()

        # Logging and training parameters
        exp.enable_wandb_logging = False
        exp.capture_video = False
        exp.eval_agent = True
        exp.eval_count = 1000
        exp.eval_frequency = 500
        exp.exp_type = None

        # Agent training parameters
        hypp.total_timesteps = int(params["total_timesteps"])
        hypp.num_epochs = int(params["num_epochs"])
        hypp.epoch_length = int(params["epoch_length"])
        hypp.gamma = float(params["gamma"])
        hypp.tau = float(params["tau"])
        hypp.log_entropy = float(params["log_entropy"])
        hypp.hidden_layers_actor = int(params["hidden_layers_actor"])
        hypp.hidden_layers_critic = int(params["hidden_layers_critic"])
        hypp.learning_rate = float(params["learning_rate"])
        hypp.learning_rate_actor = float(params["learning_rate_actor"])
        hypp.learning_rate_critic = float(params["learning_rate_critic"])
        hypp.learning_rate_model = float(params["learning_rate_model"])
        hypp.real_img_ratio = float(params["real_img_ratio"])
        hypp.display_evaluation = False
        hypp.plot_training = False
        hypp.update_param_frequency = int(params["update_param_frequency"])
        hypp.model_train_frequency = int(params["model_train_frequency"])
        hypp.rollout_schedule = [20, 150, 1, 5]
        hypp.action_prior = 'uniform'
        hypp.buffer_size = 100000
        hypp.kstep = int(params["kstep"])
        hypp.num_rollouts = int(params["num_rollouts"])
        hypp.num_ensembles = int(params["num_ensembles"])
        hypp.hidden_dim = int(params["hidden_dim"])
        hypp.batch_size = int(params["batch_size"])
        hypp.start_learning = int(params["start_learning"])
        hypp.train_frequency = int(params["train_frequency"])
        hypp.rollout_start = int(params["rollout_start"])
        hypp.max_ent_coef = float(params["max_ent_coef"])

        # Update run name
        exp.run_name = f"{exp.env_id}__{exp.exp_name}__{exp.seed}__{datetime.now().strftime('%y%m%d_%H%M%S')}"
        exp.video_path = params["video_path"]

        # Store the updated parameters
        self.exp = exp
        self.hypp = hypp
        self.env_config = env_config

    def run_training(self):
        """_summary_
        Runs the MBPO training process.

        Initiates the training using the MBPO algorithm, passing the experiment and hyperparameters for training.
        
        Returns:
            bool: Returns True when the training is complete.
        """
        carb.log_info("Running Simulation with Parameters")
        print("Experiment: ", self.exp)
        print("Hyper Parameters: ", self.hypp)
        
        trainer = mbpo(self.exp, self.hypp,self.env_config)
        trainer.train()

        return True
