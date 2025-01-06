import gymnasium
import numpy as np
#from isaacgym import gymapi
import numpy as np
from typing import Dict, Any, Tuple, List, Set
from gymnasium import spaces
import threading
from mbpo_utils import VertifarmEnv_helper_fns as env_helper_fns
import time
import carb
import os
from mbpo_utils import VertifarmEnv_helper_fns
import carb
from isaacsim import SimulationApp
import argparse
import sys
import os
from pathlib import Path
from omni_rl_bridge import OmniRosBridge
import multiprocessing
#Global Isaac Sim Settings
class Vertifarm( gymnasium.Env):
    def __init__(self):
        #VecTask.__init__(config = cfg,rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless,virtual_screen_capture = virtual_screen_capture, force_render=force_render)
        super().__init__()
        """_summary_

        Base States:
            Position (3): x,y,z
            Orientation (4): Represented as a quaternion qx ,qy,qz,qw
            Linear Velocity (3): vx,vy,vz
            Angular Velocity (3): wx,wy,wz
            Joint Angles (4): Positions for all wheel joints.
            Joint Velocities (4): Velocities for wheel joints.
            Total for base = 21 values
        Manipulator Joint States:
            Assuming the manipulator has 6DOF:

            Joint Angles (6): Positions for all joints.
            Joint Velocities (6): Velocities for all joints.
            Total for manipulator = 12

        Action Space (Control Commands)
            For a simple mobile manipulator, the actions would include:

            Base Commands (Twist commands):

                Linear Velocity (3): 3 values
                Angular Velocity (3): 3 values
                Total for base: 6 values

            Manipulator Commands (Joint Angle Changes):

                Assume control of 6 joints: 6 values
                Total for manipulator: 6 values

        """
        current_file = Path(__file__)  # Path object of the current file
        parent_directory = current_file.parent
        self.config = os.path.join(parent_directory,'config/vertifarm_env.yaml')
        #helper functions
        self.helper_fns = env_helper_fns.helper_fns()
        #load environemnt config
        if type(self.config) == str or type(self.config) == os.PathLike:
            self.env_config = self.helper_fns.load_config(self.config)
        else :
            self.env_config = self.config
        #define spaces
        obs_space = self.env_config['environment']['observation_space']
        action_space = self.env_config['environment']['action_space']
        # Convert string values to float(-inf) and float(inf)
        obs_low =  obs_space['low']
        obs_high = obs_space['high']
        action_low = obs_space['low']
        action_high = obs_space['high']
        #shape of observations and actions
        self.num_obs = np.prod(obs_space['shape'])
        self.num_actions = np.prod(action_space['shape'])
        self.observation_space = spaces.Box(np.ones(self.num_obs) * obs_low, np.ones(self.num_obs) * obs_high,dtype=np.float32)
        self.action_space = spaces.Box(np.ones(self.num_actions) *  action_low, np.ones(self.num_actions) * action_high,dtype=np.float32)
        self.observations = np.zeros(shape=(self.observation_space.shape))
        self.actions = np.zeros(shape=(self.action_space.shape))
        #some flags 
        self.rclpy_initialized = True
        self.sim_running = False
        #number of environments
        self.num_environments = 1
        # bridge to isaac sim
        self.done = False
        self.truncated = False
        self.goal = []
        self.single_observation_space = self.observation_space
        self.single_action_space = self.action_space
        self.omni_bridge = OmniRosBridge()
        current_script_dir = os.path.dirname(os.path.abspath(__file__))

        # Construct the path to the parent directory
        parent_dir = os.path.abspath(os.path.join(current_script_dir, '..'))

        # Reference a file in the parent directory
        self.omni_bridge_file = os.path.join(current_script_dir, 'omni_rl_bridge.py')
        #change it according to the current
        self.python_file_path = "/media/ashik/robotics/omni/library/isaac-sim-4.1.0/python.sh"

    def reset(self,seed=24, options=None):
        """_summary_
        Resets the Simulation state and restarts it
        Returns:
            init_obs (list): Initial Observation
        """
        # Reset the environment (robot pose, state, etc.)
        # TODO: Add arbitary or localization dictated starting position
        # for now we start from same position for simplicity.
        
        print("Vertifarm Environment Reset")
        init_obs =  self.helper_fns.terminate_sim()

        self.omni_bridge.main()
        print("Vertifarm Environment Reset")
        self.helper_fns.set_sim_status()
       
        return init_obs, {}

    def step(self, action):
        """_summary_
        Take one step into the environment
        Args:
            action (list): Curren action to take

        Returns:
            observations (list): new observation 
            reward (list): new reward
            done (bool): episde termination flag
            {} (dict) :Gym compatible info (Optional)   
        """
        # Send action to Isaac Sim, simulate, and return the next state
        #initialize the simulation
        print("Simulation Stepping Step")
        self.helper_fns.reset_stop_event()
        action_thread = threading.Thread(
            target=self.helper_fns.send_actions, 
            args=(action,),
            daemon=True
        )
        obs_thread = threading.Thread(
            target=self.helper_fns.compute_observations, 
            args=(),
            daemon=True
        )
        action_thread.start()
        obs_thread.start()
        self.omni_bridge.main()
                #initialize the observation computation thread
        
        action_thread.join()
        
        self.helper_fns.stop()
        obs_thread.join()
        print("Simulation step done")
        #compute observations
        self.observations = self.helper_fns.get_observations()
        print("Simulation step done, " , self.observations)
        #calculate reward
        reward = self.helper_fns.reward_function(self.observations)
        #see if it is done or not 
        self.done = False
        #join thread
        return self.observations, reward, self.done, self.truncated,{}
    
    def render(self):
        """_summary_
        Render Simulation
        """
        # Render the current environment (optional)
        self.helper_fns.start_subprocess(self.python_file_path, self.omni_bridge_file)
        

    def close(self):
        """_summary_
        Close Simulations
        Returns:
          final_obs (list): Final Observation 
        """
        # Clean up the environment
        print("Closing simulation")
        self.omni_bridge.terminate_simulation()
        
        final_obs =  self.helper_fns.terminate_sim()
        return final_obs


    


