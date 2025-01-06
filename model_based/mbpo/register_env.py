# register_env.py
from gymnasium.envs.registration import register

# Register the custom environment
register(
    id='VertifarmEnv-v0',  
    entry_point='VertifarmEnv.Vertifarm',  # Path to the environment class
)