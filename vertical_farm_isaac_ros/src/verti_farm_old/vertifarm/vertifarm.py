import numpy as np
import os
import torch
import isaacgymenvs


#from isaacgymenvs.tasks.base.vec_task import VecTask
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from tf_transformations import quaternion_matrix
from geometry_msgs.msg import Twist
from omni_rl_bridge import OmniRosBridge
import argparse
import sys
import subprocess
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--isaac_python", type=str, default="/media/ashik/robotics/omni/library/isaac-sim-4.1.0/python.sh",help="location of isaac sim python bash file.")
parser.add_argument("-b", "--isaac_bridge", type=str, default="omni_rl_brdige.py",help="location of isaac sim python bash file.")

args, _ = parser.parse_known_args()
class ObservationSubscriber(Node):
    def __init__(self):
        super().__init__("Subscriber_Vertifarm")
        self.joint_states_sub = self.create_subscription(JointState,"joint_states",self.joint_states_callback,10)
        self.odom_subscription = self.create_subscription(Odometry, "odom",self.odom_callback,10)
        self.sim_subscription = self.create_subscription(String, "sim_state",self.state_callback,10)

        self.observations = np.zeros(shape=(self.observation_space.shape))
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
        position = msg.position
        velocity = msg.velocity
        effort = msg.effort
        self.observations[0:13] = position
        self.observations[13:26] = velocity
        self.observations[26::36] = effort
        self.get_logger().info(f'Joint Names: {msg.name}')
        self.get_logger().info(f'Joint Positions: {msg.position}')
        self.get_logger().info(f'Joint Velocities: {msg.velocity}')
        self.get_logger().info(f'Joint Efforts: {msg.effort}')

    def state_callback(self, msg):
        self.get_logger().info(f'Simulation Status: {msg.data}')
        self.sim_running = msg.data


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
        
        self.observations[36:39] = position_vector
        self.observations[39:42]= orientation_vector
        self.observations[42:45]= linear_velocity_vector
        self.observations[45:48]= angular_velocity_vector
        self.get_logger().info(f'Linear Velocity Vector: {linear_velocity_vector}')
        self.get_logger().info(f'Angular Velocity Vector: {angular_velocity_vector}')



class Vertifarm( Node):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg
        #VecTask.__init__(config = cfg,rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless)
        super.__init__("Node")
        num_obs = 49
        num_actions = 36 

        self.num_environments = 1

        self.joint_command = self.create_publisher(JointState,"joint_command",10)
        self.cmd_vel = self.create_publisher(Twist, "cmd_vel",10)
        
        self.observations = np.zeros(shape=(self.observation_space.shape))
        self.actions = np.zeros(shape=(self.action_space.shape))

    def create_sim(self,):
        # setup simulation 
        #setup call python script brdige 
        #initial pos etc 
        result = subprocess.run([args.isaac_python,args.isaac_bridge], capture_output=True, text=True)

        if self.sim_running:
            rclpy.init()
            
            self.compute_observations()
            self.compute_reward()
        raise NotImplementedError()


    def compute_reward(self, actions):
        # must define in a way that if the step is in the actual trajectory
        #then reward, else penalty. 
        return 0
        #raise NotImplementedError()

    def compute_observations(self):
        #use instead of step
        self.subscriber = ObservationSubscriber()
        rclpy.spin_once(self.subscriber)
        if self.subscriber.sim_running:
            if self.subscriber.state_sub_flag and self.subscriber.odom_sub_flag:
                self.observations = self.subscriber.observations
                
            else:
                self.get_logger().info("All observations not published. Using old observation")
            self.subscriber.destroy_node()
            rclpy.shutdown()

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
    if __name__ == "__main__":
        create_sim()