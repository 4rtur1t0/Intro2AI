from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_vec_env
import multiprocessing

if __name__ == "__main__":
    # 1. Create 4 parallel environments to speed up training
    # Note: Use "CarRacing-v2" if your gymnasium version is older
    env_id = "CarRacing-v3"

    # Dynamically detect available CPU cores
    num_cores = multiprocessing.cpu_count()
    # Use one less than max cores so your PC doesn't freeze completely
    optimal_envs = max(1, num_cores - 1)
    print('Training with num_cores: ', optimal_envs)
    env = make_vec_env(env_id, n_envs=optimal_envs, seed=42)

    # 2. Stack 4 frames so the agent can perceive motion/speed
    env = VecFrameStack(env, n_stack=4)

    # 3. Instantiate the PPO Agent with a CNN Policy
    model = PPO(
        "CnnPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,  # Number of steps to run per environment per update
        batch_size=64,  # Minibatch size
        n_epochs=10,  # Number of epochs when optimizing the surrogate loss
        gamma=0.99,  # Discount factor
        verbose=1,
        tensorboard_log="./ppo_car_racing_tensorboard/"
    )

    # 4. Train the model (CarRacing usually requires ~1,000,000 steps to drive perfectly)
    print("Starting training... This will take a while.")
    model.learn(total_timesteps=2_000_000, progress_bar=True)

    # 5. Save the trained weights
    model.save("ppo_car_racing_agent_2_000_000")
    print("Model saved successfully!")