from gym.envs.registration import register
register(
    id="simple_env/simple_env-v0",
    entry_point="simple_env.envs:simple_env",
)
