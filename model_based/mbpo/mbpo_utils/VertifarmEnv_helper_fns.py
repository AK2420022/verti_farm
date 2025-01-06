import numpy as np
import os
from gymnasium import spaces
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import yaml
import subprocess
import os
import threading
import signal
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy,QoSDurabilityPolicy,QoSLivelinessPolicy,QoSPolicyEnum
from rclpy.duration import Duration
import time
import yaml
import threading
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
"""
Here is a list of important helper functions in the `helper_fns` class (in no particular order). You can refer to this as you get more familiar with the class:

1. `load_config` - loads a configuration file from a given path.
2. `initialize_rclpy` - initializes the ROS 2 client library (`rclpy`).
3. `shutdown_rclpy` - shuts down the ROS 2 client library (`rclpy`).
4. `spin_thread` - spins a ROS 2 node in a separate thread.
5. `create_sim` - creates and launches the Isaac Sim process for simulation.
6. `terminate_sim` - terminates the running simulation and returns initial observations.
7. `start_in_thread` - starts a specified function in a background thread.
8. `send_actions` - sends joint and wheel actions to Isaac Sim through ROS 2 topics.
9. `set_sim_status` - subscribes to the simulation status and updates the simulation state.
10. `set_goal` - sets a new goal for the robot.
11. `get_goal` - retrieves the current goal of the robot.
12. `select_dynamic_goal` - generates a new goal based on the current state.
13. `reward_function` - calculates a reward based on the distance to the goal.
14. `calculate_distance_to_goal` - computes the negative distance to the goal for reward calculation.
15. `calculate_progress_towards_goal` - calculates reward for progress made towards the goal.
16. `compute_observations` - computes the robot's observations by subscribing to relevant topics.
17. `get_observations` - retrieves the current observations of the robot.
"""
class helper_fns():
    """_summary_
    Helper functions to assist in simulating the Isaac Sim environment
    and facilitating RL Gym and reinforcement learning algorithms.

    Methods:
        load_config(config_path): Loads the configuration file in Python.
        initialize_rclpy(rclpy_initialized): Initializes the rclpy library.
        shutdown_rclpy(rclpy_initialized): Shuts down the rclpy library.
        spin_thread(node, stop_thread): Spins a ROS node once in a thread.
        create_sim(isaac_python, omni_file): Creates the simulation as a separate process if necessary.
        terminate_sim(): Terminates the running simulation and resets observations.
        start_in_thread(func): Starts a new thread with the specified function.
        send_actions(actions): Sends joint state and wheel actions to Isaac Sim.
        set_sim_status(): Reads the simulation status from a ROS topic and updates it.
        set_goal(current_goal): Sets a new goal for the simulation.
        get_goal(): Retrieves the current goal.
        select_dynamic_goal(state): Calculates a new goal by adding random values.
        reward_function(current_state): Computes a reward based on the current state.
        calculate_distance_to_goal(robot_position): Calculates the negative distance to the goal.
        calculate_progress_towards_goal(current_position): Computes progress reward based on goal proximity.
        compute_observations(sim_running): Subscribes to ROS topics and gathers observations.
        get_observations(): Retrieves the current observations.
    """
    def __init__(self):
        """_summary_
        Initialize helper functions
        """
        self.obs = np.array([0.8704395294189453, -192.46325853890687, 19.12920532596849, #position x,y,z
                    0.0,0.0,0.0,0.0, #Orientation of base x,y,z,w
                    0.0,0.0,0.0, #linear velocity x,y,z
                    0.0,0.0,0.0, #angular velocity x,y,z,w
                    #shoulder_pan_joint, LR, LF, FR, RR, shoulder_lift_joint
                    #elbow_joint,wrist_1_joint,wrist_2_joint,wrist_3_joint
                    -0.0019, 0.0001, 1.2331, 1.1682, 0.0001, 0.0004, #post
                    0.0001, 0.0006, 0.0313, 0.0532,
                    -0.0006,0.0000,0.0000,0.0000,0.0000,0.0000,
                    0.0000,0.0000,0.0000,0.0000])
        self.sim_running = False
        self.send_action_flags = False
        self.rclpy_initialized = False
        self.goal = torch.zeros((3,1), device=device)
        self.previous_position = torch.zeros((3,1), device=device)
        self.stop_event = threading.Event() 
        self.stop_compute_observations = False

    def load_config(self,config_path):
        """_summary_
        Loading the configuration file in python
        Args:
            config_path (yaml): path to the configuration file

        Returns:
            config: loaded configuration
        """
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    
    def initialize_rclpy(self):
        """_summary_
        Initialize rclpy
        Args:
            rclpy_initialized (bool): Check if rclpy has been initialized
        """
        if not self.rclpy_initialized:
            print("Initializing rclpy...")
            self.rclpy_initialized = True
            rclpy.init()
            

    def shutdown_rclpy(self):
        """_summary_
        Shutdown rclpy
        Args:
            rclpy_initialized (bool): Check if rclpy has been initialized
        """
        if self.rclpy_initialized:
            print("rclpy shutdown...")
            rclpy.shutdown()
            self.rclpy_initialized = False
            

    def spin_thread(self,node, stop_thread):
        """_summary_
        Spinning a ros node once
        Args:
            node (Ros2 Node): The Ros node to spin once
            stop_thread (bool): Ensure thread is running
        """
        while rclpy.ok() and not stop_thread.is_set():
            time.sleep(0.2) 
            rclpy.spin_once(node, timeout_sec= 10)

    def create_sim(self,isaac_python,omni_file):
        """_summary_
        Create the simulation as a seperate process if necessary.
        Args:
            isaac_python (string): path to the python.sh launcher of Isaac Sim
            omni_file (string): Name of the Omniverse Isaac Sim bridge file
        """
        # setup simulation 
        #setup call python script brdige 
        #initial pos etc 
        if self.sim_running:
            print("Another simulation running. Please close it first")
        else:
            ros_distro = os.getenv('ROS_DISTRO')
            command = isaac_python + " " + omni_file
            sim_process = subprocess.Popen([
                'gnome-terminal', '--disable-factory', '--', 'bash', '-c',
                f'source /opt/ros/{ros_distro}/setup.bash && {command}; exec bash'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            sim_pgid = os.getpgid(sim_process.pid)
            
    def terminate_sim(self,):
        """_summary_
        Terminate Running simulation
        Returns:
           init_obs (list): List of default observation 
        """
        print("Terminating simulation")

        try:
            # Kill the entire process group
            os.killpg(self.sim_pgid, signal.SIGKILL)
            print(f"Process group {self.sim_pgid} terminated.")
        except Exception as e:
            print(f"Failed to terminate the gnome-terminal: {e}")
        
        self.sim_running = False
        init_obs = [0.8704395294189453, 0.46325853890687, 19.12920532596849, #position x,y,z
                    0.0,1.0,0.0,0.0, #Orientation of base x,y,z,w
                    0.0,0.0,0.0, #linear velocity x,y,z
                    0.0,0.0,0.0, #angular velocity x,y,z,w
                    #shoulder_pan_joint, LR, LF, FR, RR, shoulder_lift_joint
                    #elbow_joint,wrist_1_joint,wrist_2_joint,wrist_3_joint
                    -0.0019, 0.0001, 1.2331, 1.1682, 0.0001, 0.0004, #post
                    0.0001, 0.0006, 0.0313, 0.0532,
                    -0.0006,0.0000,0.0000,0.0000,0.0000,0.0000,
                    0.0000,0.0000,0.0000,0.0000]
        return init_obs

    def start_in_thread(self,func):
        """_summary_
        Start a new thread with the given function
        Args:
            func (python function): The function to launch
        """
        # Create a new thread and pass the function you want to run in it
        thread = threading.Thread(target=func)
        thread.daemon = True  # Optional: makes sure the thread will exit when the main program exits
        thread.start() 

    def start_subprocess(self,executor,file):
        terminal_command = "gnome-terminal"
        # Run the .sh script as a subprocess
        subprocess.run([terminal_command, '--', 'bash', '-c', f"{executor} {file}; exec bash"])

    
    def wait_for_simulation(self, state_subscriber, timeout=30):
        """Wait until the simulation is running, or timeout."""
        start_time = time.time()
        executor = rclpy.executors.MultiThreadedExecutor()
        executor.add_node(state_subscriber)
        #self.initialize_rclpy()
        try:
            while not state_subscriber.get_sim_running():
                if time.time() - start_time > timeout:
                    raise TimeoutError("Simulation did not start within the timeout period.")
                executor.spin_once(timeout_sec=10)
                print("Waiting for simulation to start...")
        finally:
            executor.remove_node(state_subscriber)


    def send_actions(self, actions):
        """Send Actions to Isaac Sim as JointState and Twist ROS2 topics."""
        print("Sending current actions")
        self.send_action_flag = False
        self.set_sim_status()

        
        #self.initialize_rclpy()

        state_subscriber = SimStateSubscriber()

        # Wait for the simulation to start
        try:
            self.wait_for_simulation(state_subscriber, timeout=10)
        except TimeoutError as e:
            print(f"Error: {e}")
            return False

        # Initialize publishers
        joint_action = JointActionPublisher()
        cmd_vel = WheelActionPublisher()

        # Set joint and wheel actions
        joint_actions = np.array(actions[:6])
        wheel_action = np.array(actions[6:12])

        joint_action.set_joint_actions(joint_actions)
        cmd_vel.set_wheel_action(wheel_action)

        # Create a MultiThreadedExecutor
        executor = rclpy.executors.MultiThreadedExecutor()
        executor.add_node(joint_action)
        executor.add_node(cmd_vel)

        try:
            # Spin the executor
            executor.spin_once(timeout_sec=1.0)
            self.send_action_flag = True
        except Exception as e:
            print(f"Error while sending actions: {e}")
            self.send_action_flag = False
        finally:
            # Clean up nodes
            executor.remove_node(joint_action)
            executor.remove_node(cmd_vel)
            joint_action.destroy_node()
            cmd_vel.destroy_node()



    def set_sim_status(self):
        """_summary_
        Read the sim status from the 
        ROS topic and update the simulation status.
        """
        if rclpy.ok():
            state_subscriber = SimStateSubscriber()
            print("SimStateSubscriber initialized successfully")
            try:
                while rclpy.ok() :#and not stop_thread.is_set():
                    print("Subscribing to simulation status")
                    rclpy.spin_once(state_subscriber, timeout_sec= 10)
                    #while True:  # Main loop
                    
                    sim_running = state_subscriber.get_sim_running()
                    self.sim_running = sim_running
                    break
            except Exception as e:
                print(f"Error in compute_observations: {e}")
            finally:
                #stop_thread.set()
                #spin_thread_instance.join()
                state_subscriber.destroy_node()
            

    def get_send_action_done_flag(self):
        return self.send_action_flag
    def set_goal(self, current_goal):
        """_summary_
        Set the new goal as the current goal
        Args:
            current_goal (list): The new current goal how
        """
        self.goal = torch.tensor([np.random.uniform(0, 10), np.random.uniform(0, 10)])  # Random goal position


    def get_goal(self):   
        """_summary_
        Get the current goal
        Returns:
            goal (list): Return Current goal
        """
        return self.goal
    
    def select_dynamic_goal(state):
        """_summary_
        Calculate new goal by adding random value. 
        TODO: Make it more robust
        Args:
            state (list): current state

        Returns:
            new_goal (list): New goal after adding random value
        """
        # A simple rule for goal selection: s  ;.g'gv
        #  'lofclvb b;n,nlnet goal far from current agent state
        new_goal = state + np.random.randn(2)  # Slightly move the goal
        return new_goal
    
    def reward_function(self, current_state):
        """_summary_
        The reward function. provides a negative reward to the goal to
        encourage the robot to move towards the goal
        Args:
            current_state (list): The current state observed

        Returns:
            reward (float or list): current reward
        """

        current_state = torch.tensor(current_state,device=device)
        print("shape: ",current_state.shape)
        print("goal: ",self.goal.shape)
        # Calculate the reward based on distance to goal
        current_position = torch.tensor(current_state[0:3],device = device)
        orientation = torch.tensor(current_state[3:6], device=device)
        
        self.update_goal_based_on_position_and_orientation_MR(current_position, orientation)
        
        # Calculate distance-based reward
        distance_reward = self.calculate_distance_to_goal(current_position)
        
        # Calculate progress-based reward
        progress_reward = self.calculate_progress_towards_goal(current_position)
        
        # Combine rewards (you can weight these as needed)
        reward = distance_reward + progress_reward  # Adjust the combination logic as per your requirements
        
        return reward.item()
    
    def update_goal_based_on_position_and_orientation_MR(self, current_position, orientation):

        goal_distance = 100.0 #cm?
        epsilon = 1e-8
        direction_vector = orientation / (torch.norm(orientation) + epsilon)

        
        # Update the goal to be a fixed distance in front of the robot
        self.goal = (current_position + direction_vector * goal_distance).to(device)
        print("Goal goal_distance, ",self.goal)
    def calculate_distance_to_goal(self, robot_position):
        """
        Calculate the negative distance to the goal (closer is better).
        
        :param robot_position: Array-like, the current coordinates of the robot [x, y].
        :return: Reward based on distance to the goal.
        """
        print("Goal goal_distance, ",robot_position)
        print("Goal goal_distance, ",self.goal)
        
        # Assuming robot_position and self.goal are PyTorch tensors
        robot_position = torch.tensor(robot_position, dtype=torch.float32, device=device)

        # Calculate the Euclidean distance
        distance = torch.norm(robot_position - self.goal)
        return -distance  # Negative because closer should be higher reward.

    def calculate_progress_towards_goal(self, current_position):
        """
        Calculate reward based on progress towards the goal.

        :param current_position: Array-like, the current coordinates of the robot [x, y].
        :return: Reward for progress made towards the goal since the last position.
        """
        
        # Initialize previous position if it's None
        if self.previous_position is None:
            self.previous_position = current_position.clone()

        # Calculate distances to the goal
        prev_distance = torch.norm(self.previous_position - self.goal)
        curr_distance = torch.norm(current_position - self.goal)

        # Progress is the decrease in distance to the goal
        progress_reward = torch.max(torch.tensor(0.0), prev_distance - curr_distance)

        # Update the previous position
        self.previous_position = current_position.clone()

        return progress_reward

    def compute_observations(self):
        """Compute the observations by subscribing to 
        the Joint State , Odometry
        Args:
            sim_running (bool): Ensure simulation is running
        """

        #self.initialize_rclpy()
        
        # Create subscribers
        subscriber = JointStateSubscriber()
        state_subscriber = SimStateSubscriber()

        # Initialize observations list and simulation flag
        observations = []
        sim_running = False
        
        # Create MultiThreadedExecutor
        executor = rclpy.executors.MultiThreadedExecutor()
        executor.add_node(subscriber)
        executor.add_node(state_subscriber)
        self.stop_compute_observations = False
        try:
            print("Simulation started. Gathering observations...")
            while True:
                time.sleep(4.0)
                # Perform a single spin to allow both subscribers to receive messages
                executor.spin_once(timeout_sec=10.0)
                
                self.obs = subscriber.get_observations()
                print("Observation current: ", self.obs)
                sim_running = state_subscriber.get_sim_running()
                print("Simulation ,",sim_running)
                if not sim_running or self.stop_compute_observations:
                    break

                
                
        except Exception as e:
            print(f"Error in compute_observations: {e}")
        finally:
            # Clean up
            executor.remove_node(subscriber)
            executor.remove_node(state_subscriber)
            subscriber.destroy_node()
            self.shutdown_rclpy()


    def stop(self):
        """Signal the thread to stop."""
        self.stop_event.set()
    def reset_stop_event(self):
        """Reset the stop event to allow the thread to run again."""
        self.stop_compute_observations = True
        self.stop_event.clear()
    def get_observations(self,):
        """_summary_
        Get current observations
        return:
            obs (list): Current observation
        """
        return self.obs



class SimStateSubscriber(Node):
    """_summary_
    Subscriber Node for simulation state
    Args:
        Node (Ros2 Node): Base class Ros2 Node
    """
    def __init__(self):
        """_summary_
        Initialize subscriber
        """
        super().__init__("SimStateSubscriber")
        self.subscriber_initialized = False 
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,  
            durability=QoSDurabilityPolicy.VOLATILE,  
            depth=10                                 
        )

        # Set Lifespan to Infinite
        qos_profile.lifespan = Duration(seconds=0)  
        # Set Deadline to Infinite
        qos_profile.deadline = Duration(seconds=0)  

        # Set Liveliness lease duration to Infinite
        qos_profile.liveliness = QoSLivelinessPolicy.AUTOMATIC
        qos_profile.liveliness_lease_duration = Duration(seconds=0)  # Infinite liveliness lease
        self.sim_state_sub = self.create_subscription(String, "sim_state", self.state_callback, 10)
        self.sim_running = False
    def state_callback(self, msg):
        """Callback for simulation state updates.
        Args:
            msg (string): Incoming Ros2 message 
        """
        self.sim_running = msg.data == "Running"
        self.get_logger().info(f'Simulation Status: {msg.data}')
        self.get_logger().info(f'Simulation Status: {self.sim_running}')
    
    def get_sim_running(self):
        """Thread-safe getter for sim_running."""
        return self.sim_running
    
class JointStateSubscriber(Node):
    """_summary_
    Subscriber Node for JointState 
    Args:
        Node (Ros2 Node): Base class for Ros2 Node
    """
    def __init__(self):
        """_summary_
        Initialize Joint State Subscriber
        """
        super().__init__("JointStateSubscriber")
        self.subscriber_initialized = False 
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,  # Reliable delivery
            durability=QoSDurabilityPolicy.VOLATILE,   # Do not store messages for late joiners
            depth=10                                 # Queue size for History (KEEP_LAST)
        )

        # Set Lifespan to Infinite
        qos_profile.lifespan = Duration(seconds=0)   # Infinite by default

        # Set Deadline to Infinite
        qos_profile.deadline = Duration(seconds=0)   # Infinite deadline

        # Set Liveliness lease duration to Infinite
        qos_profile.liveliness = QoSLivelinessPolicy.AUTOMATIC
        qos_profile.liveliness_lease_duration = Duration(seconds=0)  # Infinite liveliness lease

        self.subscription = self.create_subscription(
            JointState,
            '/isaac_joint_states',
            self.joint_state_callback,
            qos_profile
        )
        self.subscriber_initialized = True
        self.get_logger().info("JointStateSubscriber initialized and subscribed to isaac_joint_states")
        self.odom_state_pub = self.create_subscription(Odometry, 'odom', self.odom_callback, 10)

        num_obs = 33
        self.obs_space = spaces.Box(np.ones(num_obs) * -np.Inf, np.ones(num_obs) * np.Inf)
        self.joint_names = []
        self.observations = np.zeros(shape=(self.obs_space.shape))
        self.state_sub_flag = False
        self.odom_sub_flag = False

        self.sim_running = True
        self._lock = threading.Lock()  # Add a lock for thread-safe access

    def joint_state_callback(self, msg: JointState):
        """_summary_
        Callback for joint state 
        Args:
            msg (JointState): Incoming Ros2 Message
        """
        print(f"Received JointState: {msg}")
        try:
            if self._lock:
                self.get_logger().info("Joint state callback triggered")
                self.state_sub_flag = True
                self.joint_names = msg.name
                position = msg.position
                velocity = msg.velocity
                self.observations[13:23] = position
                self.observations[23:33] = velocity
                self.get_logger().info(f'Joint Names: {msg.name}')
                self.get_logger().info(f'Joint Positions: {msg.position}')
                self.get_logger().info(f'Joint Velocities: {msg.velocity}')
        except Exception as e:
            self.get_logger().error(f"Error in joint_state_callback: {str(e)}")



    def odom_callback(self, msg):
        """_summary_
        Callback for Odometry 
        Args:
            msg (Odom): Incoming Ros2 message
        """
        print(f"Received Odom: {msg}")
        
        # Extract position
        position = msg.pose.pose.position
        self.observations[0:3] = np.array([position.x, position.y, position.z])
        
        # Extract orientation
        orientation = msg.pose.pose.orientation
        self.observations[3:7] = np.array([orientation.x, orientation.y, orientation.z, orientation.w])
        
        # Extract linear velocity
        linear_velocity = msg.twist.twist.linear
        self.observations[7:10] = np.array([linear_velocity.x, linear_velocity.y, linear_velocity.z])
        
        # Extract angular velocity
        angular_velocity = msg.twist.twist.angular
        self.observations[10:13] = np.array([angular_velocity.x, angular_velocity.y, angular_velocity.z])

        # Print the extracted data
        self.get_logger().info(f"Position: {self.observations[0:3]}")
        self.get_logger().info(f"Orientation: {self.observations[3:7]}")
        self.get_logger().info(f"Linear Velocity: {self.observations[7:10]}")
        self.get_logger().info(f"Angular Velocity: {self.observations[10:13]}")

    def get_state_sub_flag(self):
        """Thread-safe getter for sim_running."""
        with self._lock:  # Acquire lock before reading
            return self.state_sub_flag

    def get_observations(self):
        """Thread-safe getter for sim_running."""
        with self._lock:  # Acquire lock before reading
            return self.observations
        
class JointActionPublisher(Node):
    def __init__(self):
        """_summary_
        Initialize Joint Action Publisher
        """
        super().__init__("JointActionPublisher")
        self.joint_command = self.create_publisher(JointState,"isaac_joint_command",10)
        self.timer = self.create_timer(0.1, self.publish_action)  
        self.joint_names = ['shoulder_pan_joint', 'LR', 'LF', 'FR', 'RR', 'shoulder_lift_joint', 'elbow_joint','wrist_1_joint','wrist_2_joint','wrist_3_joint']
        self.joint_positions = np.zeros(shape=(len(self.joint_names)))

    def publish_action(self):
        """_summary_
        Publish Joint Action to Ros topic
        """
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        self.get_logger().info(f'Publishing joint command: {self.joint_positions}')
        msg.position  = self.joint_positions.tolist()
        #first position control
        #msg.velocity = [0.2,0.2,0.2, float('nan'), float('nan'), float('nan'), float('nan'),0.2,0.2,0.2,0.2,0.2,0.2]
        self.joint_command.publish(msg)
        self.get_logger().info(f'Publishing joint command: {msg.position}')

    def set_joint_actions(self,actions):
        """_summary_
        Set new Joint Action
        Args:
            actions (list): New Joint Action
        """
        self.joint_positions = actions

class WheelActionPublisher(Node):
    """_summary_
    Initialize Wheel Action Publisher
    Args:
        Node (Ros2 Node): Base class for Ros2 Node
    """
    def __init__(self):
        """_summary_
        Intialize Wheel Action Publisher
        """
        super().__init__("WheelActionPublisher")
        self.cmd_vel = self.create_publisher(Twist, "cmd_vel",10)
        # timer
        self.timer = self.create_timer(0.1, self.publish_action)  
        self.wheel_action = np.zeros(shape=(6))

    def publish_action(self):
        """_summary_
        Publish Wheel action to Ros Topic
        """
        msg = Twist()
        
        msg.linear.x = float(self.wheel_action[0])
        msg.linear.y = float(self.wheel_action[1])
        msg.linear.z = float(self.wheel_action[2])

        msg.angular.x = float(self.wheel_action[3])
        msg.angular.y = float(self.wheel_action[4])
        msg.angular.z = float(self.wheel_action[5])

        self.cmd_vel.publish(msg)

        self.get_logger().info(f'Publishing cmd_vel: linear.x={msg.linear.x}, angular.z={msg.angular.z}')

    def set_wheel_action(self,actions):
        """_summary_
        Set new wheel action
        Args:
            actions (list): New wheel action
        """
        self.wheel_action = actions


