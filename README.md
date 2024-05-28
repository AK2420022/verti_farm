# rl_mine
A repository for simpler implementations of different RL algorithms 
## Note
Please note that this repository is under construction, the notes on using the repository will be updated soon.
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

## Temproary workaround to support usage
 There are two different conda environment.yml files provided for each mfrl and mbrl algorithms seperately. Depending on the algorithm the irrescpective conda environment can be installed and used. 
 ### Example usage
 #### Install at default location
 '''
 conda env create -f environment.yml --prefix /location/to/install/env-name
 '''
 #### Specify the install location
 '''
 conda env create -f environment.yml
 '''
 