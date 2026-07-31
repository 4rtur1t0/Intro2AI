from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import multiprocessing

if __name__ == "__main__":
    env_id = "BipedalWalker-v3"

    # Dynamically detect available CPU cores
    num_cores = multiprocessing.cpu_count()
    # Use one less than max cores so your PC doesn't freeze completely
    optimal_envs = max(1, num_cores - 1)
    print('Using parallel environments: ', optimal_envs)

    # Create 4 parallel environments for stable and accelerated data gathering
    env = make_vec_env(env_id, n_envs=optimal_envs, seed=42)

    # Initialize PPO with an MlpPolicy (Multi-Layer Perceptron for feature vectors)
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,  # Number of steps per env before optimizing
        batch_size=64,  # Minibatch size
        n_epochs=10,  # Number of gradient epochs per update
        gamma=0.99,  # Discount factor
        gae_lambda=0.95,  # Factor for trade-off of bias vs variance for GAE
        verbose=1,
        tensorboard_log="./ppo_bipedal_walker_tensorboard/"
    )

    # Train the agent. It generally learns to walk securely around 500k to 1M steps.
    print("Training started...")
    model.learn(total_timesteps=2_000_000, progress_bar=True)

    # Save the trained model weights
    model.save("ppo_bipedal_walker_agent_2_000_000")
    print("Model saved successfully!")