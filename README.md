[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-360/)

# The Vertifarm Project

## Note
Please note that this repository is under construction, the notes on using the repository will be updated soon. Check **vpi branch** for the latest updates.

# VertiFarm project
The vertiFarm project is aimed to accelerate production in vertical farming and similar environments with mobile manipulator robot infrastructure.

##  The project is under contruction. The modules are as follows
## vertical_farm_isaac_ros
   The IsaacSim environment customized for Multi Agent vertical farming, equipped with mobile manipulators to perform various tasks (current showcase focuses on visual pathalogical inspection of leaves). Current environment is fully connected with ros2 and aims to provides the necessary packages to establish communication, learn different tasks and navigation through MBRL and other Machine Learning menthods. 
#### Current Experiments - 
 
 ##### Model based RL for mobile robot navigation and planning using visual states and latent models
 ##### Domain randomization and procedural generation for better learning and optimization for simulation scenarios (sim2real)
 ##### Visual pathalogical inspection of tomatoe plants
 
### Architecture

![alt text](docs/arch.png)

## Sample Environment
![alt text](docs/env.png)

![alt text](docs/tif.png)

https://github.com/AK2420022/rl_mine/assets/19958594/0487f15a-92ec-48b9-af7d-16069d94def6
## RL Algorithms 
   #### Deep learning based Model Free RL algorithms(The algorithms require gymnasium >=0.26 )
   - DDPG
   - DQN / DDQN
   - PPO
   - TD3
   - SAC
   #### Model based RL algorithms (MBRL) (The algorithms require gymnasium v1.0.0 alpha 2)
   - MBPO experimentation with Probabilistic Neural Network ensemble as world model and VertiFarm as the true Environment
   - DYNAQ / DYNAQ+
   #### Hingsight Experience Replay(HER) algorithms(The algorithm requires gymnasium >=0.26 )
   - With DDPG

## Temproary workaround to support usage
 There are two different conda environment.yml files provided for each mfrl and mbrl algorithms seperately. Depending on the algorithm the irrescpective conda environment can be installed and used. 
 ### Example usage
 #### Specify the install location
 ```
 conda env create -f environment.yml --prefix /location/to/install/env-name
 ```
  #### Install at default location
 ```
 conda env create -f environment.yml
 ```
 #### Setting up the Isaac Sim Environment 
 Extract the following packages and then open the example environment farmer2.usd in Isaac sim - 
 - vertical_farm_isaac_ros/src/vertifarm/omni_assets/TIF/Collected_farmer/farmer2.tar.xz
 - vertical_farm_isaac_ros/src/vertifarm/omni_assets/TIF/Collected_farmer/TIF/vert/t1.tar.xz
 #### Setting up moveit support and ROS2 packages 
 The current version supports **ROS2 Humble** and moveit2
 You can follow the general [instructions](https://docs.ros.org/en/eloquent/Tutorials/Creating-Your-First-ROS2-Package.html#build-a-package) to build the relevant packages inside of src folder witin the **vertical_farm_isaac_ros** workspace . 
   
The following resources have been an inspiration for the RL implementations - 
-  Reinforcement Learning and Learning Based Control, SoSe 2022 - Prof. Sebastin Trimpe - RWTH Aachen University, Germany
-  [DLR stable-baselines3](https://github.com/DLR-RM/stable-baselines3/tree/master)
