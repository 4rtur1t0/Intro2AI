import gymnasium as gym
from stable_baselines3 import PPO

# Initialize the environment with human rendering enabled
env = gym.make("BipedalWalker-v3", render_mode="human")

# Load the saved PPO agent
model = PPO.load("ppo_bipedal_walker_agent_2_000_000.zip")

obs, info = env.reset()
while True:
    # Use deterministic=True for execution evaluation to avoid random exploratory movements
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()