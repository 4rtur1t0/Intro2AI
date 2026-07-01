import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import ale_py  # 1. Import the Arcade Learning Environment backend



if __name__ == "__main__":
    # 2. Explicitly register the ALE environments with Gymnasium
    gym.register_envs(ale_py)

    # 3. Create 8 parallel Atari environments
    env_id = "ALE/Breakout-v5"


    # Load environment with human rendering enabled
    env = make_atari_env(env_id=env_id, n_envs=1, env_kwargs={"render_mode": "human"})
    env = VecFrameStack(env, n_stack=4)

    model = PPO.load("ppo_breakout_agent_5_000_000", env=env)

    obs = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=False) # False allows natural play style
        obs, rewards, dones, infos = env.step(action)