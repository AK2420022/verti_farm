import numpy as np
import os
import torch
import isaacgymenvs
from typing import Dict, Any, Tuple, List, Set
from gym import spaces
#from isaacgymenvs.tasks.base.vec_task import VecTask
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from tf_transformations import quaternion_matrix
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import argparse
import sys
import yaml
import subprocess
import os
import glob
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped
import threading
import signal
current_directory = os.path.dirname(os.path.abspath(__file__))
print("sdsd: ," , current_directory)
omni_file = os.path.join(current_directory, "omni_rl_bridge.py")

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--isaac_python", type=str, default="/media/ashik/robotics/omni/library/isaac-sim-4.1.0/python.sh",help="location of isaac sim python bash file.")
parser.add_argument("-b", "--isaac_bridge", type=str, default="omni_rl_brdige.py",help="location of isaac sim python bash file.")

args, _ = parser.parse_known_args()
class JointStateSubscriber(Node):
    def __init__(self):
        super().__init__("JointStateSubscriber")
        self.joint_states_sub = self.create_subscription(JointState,"joint_states",self.joint_state_callback,10)
        self.odom_state_pub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10)   
        self.sim_state_sub = self.create_subscription(String,"sim_state",self.state_callback,10)
     
        self.num_obs = 52
        self.obs_space = spaces.Box(np.ones(self.num_obs) * -np.Inf, np.ones(self.num_obs) * np.Inf)
        self.joint_names = []
        self.observations = np.zeros(shape=(self.obs_space.shape))
        self.state_sub_flag = False
        self.odom_sub_flag = False
        self.sim_running = False
        
    def joint_state_callback(self, msg):
        """_summary_
            1. ur_arm_shoulder_pan_joint
            2. ur_arm_shoulder_lift_joint
            3. dummy_base_revolute_z_joint
            4. LR
            5. LF
            6. FR
            7. RR
            8. ur_arm_elbow_joint
            9. dummy_base_prismatic_y_joint
            10. ur_arm_wrist_1_joint
            11. dummy_base_prismatic_x_joint
            12. ur_arm_wrist_2_joint
            13. ur_arm_wrist_3_joint
        Args:
            msg (_type_): _description_
        """
        self.state_sub_flag = True
        self.joint_names = msg.name
        position = msg.position
        velocity = msg.velocity
        effort = msg.effort
        self.observations[0:13] = position
        self.observations[13:26] = velocity
        self.observations[26:39] = effort
        self.get_logger().info(f'Joint Names: {msg.name}')
        self.get_logger().info(f'Joint Positions: {msg.position}')
        self.get_logger().info(f'Joint Velocities: {msg.velocity}')
        self.get_logger().info(f'Joint Efforts: {msg.effort}')
        """
        class SimSubscription(Node):
            def __init__(self):
                super().__init__("SimSubscription")
                self.sim_subscription = self.create_subscription(String, "sim_state",self.state_callback,10)
                self.sim_running = False"""
    def state_callback(self, msg):
        self.get_logger().info(f'Simulation Status: {msg.data}')
        self.sim_running = msg.data
        """
    class OdomSubscription(Node):
        def __init__(self):
            super().__init__("OdomSubscription")
            self.odom_subscription = self.create_subscription(Odometry, "odom",self.odom_callback,10)
            self.num_obs = 13
            self.obs_space = spaces.Box(np.ones(self.num_obs) * -np.Inf, np.ones(self.num_obs) * np.Inf)

            self.observations = np.zeros(shape=(self.obs_space.shape))
            self.odom_sub_flag = False
    """
    def odom_callback(self, msg):
        self.odom_sub_flag = True
        # Extract position as a vector
        position_vector = [msg.pose.pose.position.x, 
                           msg.pose.pose.position.y, 
                           msg.pose.pose.position.z]
        self.get_logger().info(f'Position Vector: {position_vector}')

        # Extract orientation (quaternion) and convert to a matrix
        orientation = msg.pose.pose.orientation
        orientation_quaternion = [orientation.x, orientation.y, orientation.z, orientation.w]
        orientation_matrix = quaternion_matrix(orientation_quaternion)
        
        # Extract the orientation vector from the rotation matrix (example: x-axis direction)
        orientation_vector = orientation_matrix[0:3, 0]
        self.get_logger().info(f'Orientation Vector (X-axis direction): {orientation_vector}')

        # Extract linear and angular velocities as vectors
        linear_velocity_vector = [msg.twist.twist.linear.x,
                                  msg.twist.twist.linear.y,
                                  msg.twist.twist.linear.z]
        angular_velocity_vector = [msg.twist.twist.angular.x,
                                   msg.twist.twist.angular.y,
                                   msg.twist.twist.angular.z]
        
        self.observations[39:42] = position_vector
        self.observations[42:45]= orientation_vector
        self.observations[45:48]= linear_velocity_vector
        self.observations[48:51]= angular_velocity_vector
        self.get_logger().info(f'Linear Velocity Vector: {linear_velocity_vector}')
        self.get_logger().info(f'Angular Velocity Vector: {angular_velocity_vector}')

class JointActionPublisher(Node):
    def __init__(self):
        super().__init__("JointActionPublisher")
        self.joint_command = self.create_publisher(JointState,"joint_command",10)
        # timer
        self.timer = self.create_timer(0.1, self.publish_action)  
        self.joint_names = ['ur_arm_shoulder_pan_joint', 'ur_arm_shoulder_lift_joint', 'dummy_base_revolute_z_joint', 'LR', 'LF', 'FR', 'RR', 'ur_arm_elbow_joint', 'dummy_base_prismatic_y_joint', 'ur_arm_wrist_1_joint', 'dummy_base_prismatic_x_joint', 'ur_arm_wrist_2_joint', 'ur_arm_wrist_3_joint']
        self.joint_positions = np.zeros(len(self.joint_names))
    def publish_action(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position  = [0.2,0.2,0.2, float('nan'), float('nan'), float('nan'), float('nan'),0.2,0.2,0.2,0.2,0.2,0.2]
        msg.velocity = [0.2,0.2,0.2, float('nan'), float('nan'), float('nan'), float('nan'),0.2,0.2,0.2,0.2,0.2,0.2]
        self.joint_command.publish(msg)
        self.get_logger().info(f'Publishing joint command: {msg.position}')

class WheelActionPublisher(Node):
    def __init__(self):
        super().__init__("WheelActionPublisher")
        self.cmd_vel = self.create_publisher(Twist, "cmd_vel",10)
        # timer
        self.timer = self.create_timer(0.1, self.publish_action)  
        self.linear_velocity = 0.01
        self.angular_velocity = 0.01
    def publish_action(self):
        msg = Twist()
        msg.linear.x = self.linear_velocity
        msg.linear.y = 0.0
        msg.linear.z = 0.0

        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = self.angular_velocity

        self.cmd_vel.publish(msg)

        self.get_logger().info(f'Publishing cmd_vel: linear.x={msg.linear.x}, angular.z={msg.angular.z}')


config_file = "/media/ashik/robotics/IsaacSim-nonros_workspaces/src/rl_mine/vertical_farm_isaac_ros/src/verti_farm/cfg/ur5.yaml"


class Vertifarm( Node):
    def __init__(self,config: Dict[str, Any] ):
        #VecTask.__init__(config = cfg,rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless,virtual_screen_capture = virtual_screen_capture, force_render=force_render)
        super().__init__("vertifarm")
        self.num_obs = 52
        self.num_actions = 36 
        self.rclpy_initialized = True
        self.num_environments = 1
        self.joint_names = []
        self.sim_running = False
        self.obs_space = spaces.Box(np.ones(self.num_obs) * -np.Inf, np.ones(self.num_obs) * np.Inf)
        self.action_space = spaces.Box(np.ones(self.num_actions) * -1, np.ones(self.num_actions) * 1)
        self.observations = np.zeros(shape=(self.obs_space.shape))
        self.actions = np.zeros(shape=(self.action_space.shape))
        self.sim_process = None
        self.shutdown_rclpy()
    def initialize_rclpy(self):
        if not self.rclpy_initialized:
            rclpy.init()
            self.rclpy_initialized = True
            print("rclpy initialized")
    def shutdown_rclpy(self):
        if self.rclpy_initialized:
            rclpy.shutdown()
            self.rclpy_initialized = False
            print("rclpy shutdown")
    def spin_thread(self,node):
        rclpy.spin(node)
    def create_sim(self,):
        # setup simulation 
        #setup call python script brdige 
        #initial pos etc 
        if self.sim_running:
            print("Another simulation running. Please close it first")
        else:
            ros_distro = os.getenv('ROS_DISTRO')
            command = args.isaac_python + " " + omni_file
            self.sim_process = subprocess.Popen([
                'gnome-terminal', '--disable-factory', '--', 'bash', '-c',
                f'source /opt/ros/{ros_distro}/setup.bash && {command}; exec bash'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.sim_pgid = os.getpgid(self.sim_process.pid)
            
    def terminate_sim(self,):
        if self.sim_process and self.sim_running:
            print("Terminating simulation")

            try:
                # Kill the entire process group
                os.killpg(self.sim_pgid, signal.SIGKILL)
                print(f"Process group {self.sim_pgid} terminated.")
            except Exception as e:
                print(f"Failed to terminate the gnome-terminal: {e}")
            
            self.sim_running = False
        else:
            print("No simulation is running.")
    
    def compute_actions(self):
        print("compute_actions")
        self.initialize_rclpy()
        #publisher thread 
        joint_action = JointActionPublisher()
        thread_1 = threading.Thread(target=rclpy.spin, args=(joint_action, ), daemon=True)
        thread_1.start()
        cmd_vel = WheelActionPublisher()
        thread_2 = threading.Thread(target=rclpy.spin, args=(cmd_vel, ), daemon=True)
        thread_2.start()
        try:
            while rclpy.ok():
                # Insert a sleep or wait mechanism if necessary to control the loop frequency
                rclpy.spin_once(joint_action, timeout_sec=0.1)
                rclpy.spin_once(cmd_vel, timeout_sec=0.1)
                break
        finally:
            joint_action.destroy_node()
            cmd_vel.destroy_node()
            self.shutdown_rclpy()
            thread_1.join()
            thread_2.join()

    def compute_reward(self, actions):
        # must define in a way that if the step is in the actual trajectory
        #then reward, else penalty. 
        return 0
        #raise NotImplementedError()
        
    def compute_observations(self):
        self.initialize_rclpy()
        # use instead of step
        # Initialize your subscriber only once
        self.subscriber = JointStateSubscriber()
        # Start the spinning in a separate thread
        spin_thread_instance = threading.Thread(target=self.spin_thread, args=(self.subscriber,))
        spin_thread_instance.start()

        try:
            while rclpy.ok():
                self.sim_running = self.subscriber.sim_running
                if self.subscriber.sim_running:
                    if self.subscriber.state_sub_flag and self.subscriber.odom_sub_flag:
                        self.observations = self.subscriber.observations
                        print("Observation current: ", self.observations)
                        break        
                    else:
                        print("All observations not published. Using old observation")
            # Insert a sleep or wait mechanism if necessary to control the loop frequency
            #rclpy.spin_once(self.subscriber, timeout_sec=0.1)

        finally:
            self.subscriber.destroy_node()
            self.shutdown_rclpy()
            spin_thread_instance.join()
        
        return self.observations

    def compute_ur5_reward(
        reset_buf, progress_buf, actions, cabinet_dof_pos,
        franka_grasp_pos, drawer_grasp_pos, franka_grasp_rot, drawer_grasp_rot,
        franka_lfinger_pos, franka_rfinger_pos,
        gripper_forward_axis, drawer_inward_axis, gripper_up_axis, drawer_up_axis,
        num_envs, dist_reward_scale, rot_reward_scale, around_handle_reward_scale, open_reward_scale,
        finger_dist_reward_scale, action_penalty_scale, distX_offset, max_episode_length
    ):
        # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, float, float, float, float, float, float, float, float) -> Tuple[Tensor, Tensor]

        # distance from hand to the drawer
        raise NotImplementedError()

    