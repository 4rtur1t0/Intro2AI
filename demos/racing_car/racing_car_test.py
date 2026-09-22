# import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_vec_env

# Load environment with human render mode so you can watch it
env = make_vec_env("CarRacing-v3", n_envs=1, env_kwargs={"render_mode": "human"})
env = VecFrameStack(env, n_stack=4)

# Load the trained model
model = PPO.load("ppo_car_racing_agent_2_000_000.zip", env=env)

obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, infos = env.step(action)

    # VecEnv automatically resets on done, no need to manually handle it