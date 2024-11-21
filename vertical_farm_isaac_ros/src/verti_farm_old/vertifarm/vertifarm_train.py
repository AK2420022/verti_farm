from vertifarm import Vertifarm
import yaml
import rclpy
config_file = "/media/ashik/robotics/IsaacSim-nonros_workspaces/src/rl_mine/vertical_farm_isaac_ros/src/verti_farm/cfg/ur5.yaml"
def main():
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    rl_device = config["rl_device"]
    sim_Device = config["sim_device"]
    graphics_Device = config["graphics_device"]
    headless = config["headless"]

    verti = Vertifarm(config, rl_device, sim_Device, graphics_Device,headless)
    verti.create_sim()
