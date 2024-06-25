# rl_mine
A repository for simpler implementations of different RL algorithms 
## Note
Please note that this repository is under construction, the notes on using the repository will be updated soon. Check **develop branch** for the latest updates.
### Algorithms implemented
 #### Deep learning based Model Free RL algorithms(The algorithms require gymnasium >=0.26 )
 - DDPG
 - DQN / DDQN
 - PPO
 - TD3
 - SAC
 #### Model based RL algorithms (The algorithms require gymnasium v1.0.0 alpha 2)
 - MBPO #under work, tuning required
 - DYNAQ / DYNAQ+
 #### Hingsight Experience Replay(HER) algorithms(The algorithm requires gymnasium >=0.26 )
 - With DDPG # under work, tuning required

# VertiFarm project (Under construction)
The vertiFarm project is aimed to accelerate vertical indoor farms with mobile robots that can navigate and perform mutiple tasks without minimal or no human effort.
##  Current relevant modules available and their brief usage
 - verti_farm : The IsaacSim environment for vertical farming, equipped with a mobile robot and mounted ur5 manipulator as the ego robot. Current environment is fully connected with ros2 and provides the necessary communication to establish, learn different tasks and navigation through RL and other menthods in ros2.
 - isaac_moveit_ur5control : Current control and trajectory planning interface for the ur5 manipulator to be controlled and manipulated in IsaacSim. 
![alt text](docs/verti_farm.png)

![alt text](docs/env.png)

Experiments for  Simulation of planning, perception and control of multi-agent systems and cobots using Model based Reinforcement Learning.


https://github.com/AK2420022/rl_mine/assets/19958594/0487f15a-92ec-48b9-af7d-16069d94def6


## Temproary workaround to support usage
 There are two different conda environment.yml files provided for each mfrl and mbrl algorithms seperately. Depending on the algorithm the irrescpective conda environment can be installed and used. 
 ### Example usage
 #### Install at default location
 ```
 conda env create -f environment.yml --prefix /location/to/install/env-name
 ```
 #### Specify the install location
 ```
 conda env create -f environment.yml
 ```
 
