import gymnasium as gym
import ale_py  # 1. Import the Arcade Learning Environment backend
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import multiprocessing

if __name__ == "__main__":
    # 2. Explicitly register the ALE environments with Gymnasium
    gym.register_envs(ale_py)

    # 3. Create 8 parallel Atari environments
    env_id = "ALE/Breakout-v5"
    # Dynamically detect available CPU cores
    num_cores = multiprocessing.cpu_count()
    # Use one less than max cores so your PC doesn't freeze completely
    optimal_envs = max(1, num_cores - 1)
    print('Using parallel environments: ', optimal_envs)
    env = make_atari_env(env_id, n_envs=optimal_envs, seed=42)

    # 4. Stack 4 frames so the agent can see the speed and direction of the ball
    env = VecFrameStack(env, n_stack=4)

    # 5. Initialize PPO with a CNN policy
    model = PPO(
        "CnnPolicy",
        env,
        learning_rate=2.5e-4,
        n_steps=128,
        batch_size=256,
        n_epochs=4,
        clip_range=0.1,
        vf_coef=0.5,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log="./ppo_breakout_tensorboard/"
    )

    # 6. Train the agent
    print("Training started... watch the tensorboard for progress!")
    model.learn(total_timesteps=5_000_000, progress_bar=True)

    # 7. Save the model
    model.save("ppo_breakout_agent_5_000_000")
    print("Model saved!")