# Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
import carb
from isaacsim import SimulationApp
import argparse
import sys
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
# enable ROS2 bridge extension
enable_extension("omni.isaac.ros2_bridge")

simulation_app.update()

import time

# Note that this is not the system level rclpy, but one compiled for omniverse
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu, JointState,PointCloud2,Image,LaserScan
from tf2_msgs.msg import TFMessage
#from Messages.msg import NitroBrdigeImage
class OmniRosBridge(Node):
    def __init__(self):
        super().__init__("omni_bridge_initialize")
        #test empty world
        self.timeline = omni.timeline.get_timeline_interface()
        assets_root_path = nucleus.get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
            simulation_app.close()
            sys.exit()
        # setup the ROS2 subscriber here
        # setup action commands to manipulator and mobile robot 
        self.cmd_vel = self.create_publisher(Twist, "cmd_vel",10)
        self.joint_command = self.create_publisher(JointState, "joint_command",10)
        # setup subscription of different sensors and states
        self.imu = self.create_subscription(Imu,"imu",self.imu_callback,10)
        self.joint_states = self.create_subscription(JointState,"joint_states",self.joint_states_callback,10)
        self.odom = self.create_subscription(Twist,"odom",self.odom_callback,10)
        self.point_cloud =  self.create_subscription(PointCloud2,"point_cloud",self.point_cloud_callback,10)
        self.stereo_left =  self.create_subscription(Image,"stereo_left",self.stereo_left_callback,10)
        self.stereo_right =  self.create_subscription(Image,"stereo_right",self.stereo_right_callback,10)
        self.tf =  self.create_subscription(TFMessage,"tf",self.tf_callback,10)
        self.scan = self.create_subscription(LaserScan,"scan",self.scan_callback,10)
        #self.ros_world.reset()
        self.wait_updates = 10
        
    def imu_callback(self, data):
        print("Imu data: " ,data)

    def joint_states_callback(self, data):
        print("Joint state data: " ,data)

    def odom_callback(self, data):
        print("odom data: " ,data)
    
    def point_cloud_callback(self, data):
        print("point_cloud data: " ,data)
    
    def stereo_left_callback(self, data):
        print("stereo left data: " ,data)
    
    def stereo_right_callback(self, data):
        print("stereo right data: " ,data)
    
    def tf_callback(self, data):
        print("tf data: " ,data)
    
    def scan_callback(self, data):
        print("scan data: " ,data)

    def setup_simulation(self):
        omni.usd.get_context().open_stage(ENV_PATH, None)
        for i in range(self.wait_updates):
            simulation_app.update()
        print("Loading stage...")
        while is_stage_loading():
            simulation_app.update()
        print("Loading Complete")
        

        simulation_app.update()
        simulation_context = SimulationContext(stage_units_in_meters=0.01)
        simulation_context.play()
        
        simulation_app.update()
        self.run_simulation(simulation_context)
    def run_simulation(self, simulation_context):
        self.timeline.play()
        while simulation_app.is_running():
            print("Simulation is running")
            simulation_context.step()
            rclpy.spin_once(self,timeout_sec = 1.0)
        # Cleanup
        self.timeline.stop()
        self.destroy_node()
        simulation_app.close()


if __name__ == "__main__":
    rclpy.init()
    
    subscriber = OmniRosBridge()
    subscriber.setup_simulation()
    