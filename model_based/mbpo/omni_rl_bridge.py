# Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
#!/usr/bin/env python3

import carb
from isaacsim import SimulationApp
import argparse
import sys
import os

# Global Isaac Sim Settings
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, default="Example_Rotary", help="Name of lidar config.")
parser.add_argument("-i", "--isaac_python", type=str, default="", help="Location of Isaac Sim Python bash file.")
args, _ = parser.parse_known_args()
CONFIG = {"renderer": "RayTracedLighting", "headless": True}
simulation_app = SimulationApp(CONFIG)

# List of topics
ENV_PATH = "/media/ashik/robotics/IsaacSim-nonros_workspaces/src/rl_mine/vertical_farm_isaac_ros/src/vertifarm/omni_assets/TIF/Collected_farmer/farmer2.usd"
parent_dir = os.path.dirname(os.path.abspath(__file__))

# Add it to sys.path
sys.path.append(parent_dir)
from mbpo_utils import VertifarmEnv_helper_fns as helper_fns

# Omni header files
import omni
from omni.isaac.core import SimulationContext
from omni.isaac.core.utils import extensions, nucleus, prims, rotations, stage, viewports
from omni.isaac.core import World
from omni.isaac.core.utils.extensions import enable_extension
from omni.isaac.core.utils.stage import is_stage_loading
import omni.kit.viewport.utility
import omni.replicator.core as rep
from omni.isaac.core import SimulationContext
from omni.isaac.core.utils import nucleus, stage
from omni.isaac.core.utils.extensions import enable_extension
from pxr import Gf

# Enable ROS2 bridge extension
enable_extension("omni.isaac.debug_draw")
enable_extension("omni.isaac.ros2_bridge")
simulation_app.update()

import time
start_time = time.time()  # Record the start time

# Note that this is not the system-level rclpy, but one compiled for Omniverse
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import threading
from rclpy.executors import MultiThreadedExecutor

class SimStatePublsiher(Node):
    """Publisher for SimulationState

    Args:
        Node (Ros2 Node): Base class Ros2 Node
    """
    def __init__(self, data):
        """Initialize the SimState publisher

        Args:
            data (String): Sim Status to publish "Running" or "Stop"
        """
        super().__init__('IsaacRosBridge')  
        self.running_pub = self.create_publisher(String, "sim_state", 10)
        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.msg = String()
        self.msg.data = data
    
    def timer_callback(self):
        """Callback function for publisher"""
        self.running_pub.publish(self.msg)
        self.get_logger().info('Publishing Sim Status: "%s"' % self.msg.data)

class OmniRosBridge():
    """The bridge to Isaac Sim and Ros2, with the RL Environment

    Args:
        None
    """
    def __init__(self):
        """Initialize the bridge"""
        self.timeline = omni.timeline.get_timeline_interface()
        self.assets_root_path = nucleus.get_assets_root_path()
        self.sim_running = False
        self.publisher = None
        self.helper = helper_fns.helper_fns()
        self.duration = 2
        self.terminate_sim = False

    def run_simulation(self):
        """Run the Isaac Sim simulation"""
        #self.helper.initialize_rclpy()
        omni.usd.get_context().open_stage(ENV_PATH, None)
        simulation_app.update()

        if not self.assets_root_path:
            carb.log_error("Could not find Isaac Sim assets folder")
            simulation_app.close()
            sys.exit()

        simulation_context = SimulationContext(
            physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0, stage_units_in_meters=0.01
        )
        simulation_app.update()
        
        simulation_context.play()
        self.sim_running = True
        # Create ROS2 publisher
        self.publisher = SimStatePublsiher("Running")
        executor = MultiThreadedExecutor()
        executor.add_node(self.publisher)
        executor_thread = threading.Thread(target=executor.spin, daemon=True)
        executor_thread.start()
        actions = np.zeros(shape=(16))
        start_time = time.time()
        step = 0
        try:
            while simulation_app.is_running() and self.sim_running:
                simulation_app.update()            
                if (time.time() - start_time) > self.duration:
                    self.terminate_simulation()
                    break
                if self.terminate_sim:
                    self.terminate_simulation()
                    break
                self.helper.send_actions(actions)
                print(f"Simulation time: {(time.time() - start_time)}")
                print(f"Simulation step: {step}")
                step += 1
        except KeyboardInterrupt:
            print("Shutting down...")
        finally:
            print("Shutting down...")
            self.cleanup_simulation(executor)

    def set_terminate_sim(self):
        """Set the termination flag"""
        self.terminate_sim = True

    def terminate_simulation(self):
        """Terminate the simulation"""
        time.sleep(0.2)
        self.sim_running = False
        self.publisher.msg.data = "Stop"

        rclpy.spin_once(self.publisher)
        #simulation_app.close()

    def cleanup_simulation(self, executor):
        """Cleanup the simulation and all threads

        Args:
            executor (thread): The thread to terminate
        """
        executor.shutdown()
        if self.publisher:
            self.publisher.destroy_node()
        #self.helper.shutdown_rclpy()
    def close_bridge(self):
        """_summary_
        close the isaac sim simulation
        """
        if self.publisher:
            self.publisher.destroy_node()
        #simulation_app.close()
    def main(self):
        """Main entry function"""
        #rclpy.init()
        self.run_simulation()


