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
import subprocess
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, default="Example_Rotary", help="Name of lidar config.")
parser.add_argument("-i", "--isaac_python", type=str, default="",help="location of isaac sim python bash file.")
args, _ = parser.parse_known_args()
CONFIG = {"renderer": "RayTracedLighting", "headless": False}
simulation_app = SimulationApp(CONFIG)
#list of topics
ENV_PATH = "/media/ashik/robotics/IsaacSim-nonros_workspaces/src/rl_mine/vertical_farm_isaac_ros/src/verti_farm/omni_assets/Collected_farmer/farmer.usd"

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

# enable ROS2 bridge extension

enable_extension("omni.isaac.debug_draw")
enable_extension("omni.isaac.ros2_bridge")
simulation_app.update()

import time

# Note that this is not the system level rclpy, but one compiled for omniverse
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
class SimStatePublsiher(Node):
    def __init__(self):
        super().__init__('IsaacRosBridge')  
        self.running_pub = self.create_publisher(String,"sim_state",10)
        timer_period = 0.1 # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
    def timer_callback(self):
        msg = String()
        msg.data = "Running"
        self.running_pub.publish(msg)
        self.get_logger().info('Publishing Sim Status: "%s"' % msg.data)
    
        
#from Messages.msg import NitroBrdigeImage
class OmniRosBridge():
    def __init__(self):
        #test empty world
        self.timeline = omni.timeline.get_timeline_interface()
        self.assets_root_path = nucleus.get_assets_root_path()
        self.sim_running  = False


    def run_simulation(self, ):
        omni.usd.get_context().open_stage(ENV_PATH, None)

        simulation_app.update()

        # Locate Isaac Sim assets folder to load environment and robot stages
        assets_root_path = nucleus.get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
            simulation_app.close()
            sys.exit()


        simulation_app.update()
        # Loading the simple_room environment
        #stage.add_reference_to_stage(    assets_root_path + "/Isaac/Environments/Simple_Warehouse/full_warehouse.usd", "/background")
        simulation_app.update()
        """
        lidar_config = args.config
        
        # Create the lidar sensor that generates data into "RtxSensorCpu"
        # Possible config options are Example_Rotary and Example_Solid_State
        _, sensor = omni.kit.commands.execute(
            "IsaacSensorCreateRtxLidar",
            path="/sensor",
            parent="/World/ridgeback/ridgeback_ur5_01/base_link/lidar",
            config=lidar_config,
            translation=(0.34, -0.04, 0.33),
            orientation=Gf.Quatd(1.0, 0.0, 0.0, 0.0),  # Gf.Quatd is w,i,j,k
        )

        hydra_texture = rep.create.render_product(sensor.GetPath(), [480, 640], name="Isaac")
        """
        simulation_context = SimulationContext(physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0, stage_units_in_meters=0.01)
        simulation_app.update()
        """
        # Create the debug draw pipeline in the post process graph
        writer = rep.writers.get("RtxLidar" + "DebugDrawPointCloud" + "Buffer")
        writer.initialize(color =[1.0,0.0,0.0,1.0],transform = [[100.0,0.0,0.0,0.0],[0.0,100.0,0.0,0.0],[0.0,0.0,100.0,0.0],[0.0,0.0,0.0,1.0]],doTransform=True)
        writer.attach([hydra_texture])

        writer = rep.writers.get("RtxLidar" + "ROS2PublishPointCloud")
        writer.initialize(topicName="point_cloud", frameId="base_link")

        writer.attach([hydra_texture])
        """
        simulation_app.update()
        self.sim_running = True
        simulation_context.play()
        publisher = SimStatePublsiher()
        while simulation_app.is_running():
            simulation_app.update()
            rclpy.spin_once(publisher)
        publisher.destroy_node()
        rclpy.shutdown()


#s
def run_bridge():
    rclpy.init()
    sim = OmniRosBridge()
    sim.run_simulation()

    
if __name__ == "__main__":
    run_bridge()