from vertifarm.verti_env import Vertifarm 
import yaml
import rclpy
import threading
config_file = "/media/ashik/robotics/IsaacSim-nonros_workspaces/src/rl_mine/vertical_farm_isaac_ros/src/verti_farm/cfg/ur5.yaml"
def main():
    rclpy.init()
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    
    vertifarm = Vertifarm(config)
    vertifarm.create_sim()
    
    observations = vertifarm.compute_observations()
    vertifarm.compute_actions()
    observations = vertifarm.compute_observations()
    print(observations)
    
    vertifarm.terminate_sim()