from collections import deque
import copy
import random
import numpy as np
import gymnasium as gym
from sklearn.neural_network import MLPRegressor

# Initialize Environment
try:
    env = gym.make("LunarLander-v3")
except Exception:
    env = gym.make("LunarLander-v2")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# 1. Initialize Neural Networks
# Scikit-Learn's MLPRegressor acts as our Q-function approximator
q_net = MLPRegressor(
    hidden_layer_sizes=(64, 64),
    activation="relu",
    solver="adam",
    learning_rate_init=0.0005,
    max_iter=1,
)

# Run a single dummy partial_fit to initialize network weights and dimensions
dummy_X = np.zeros((1, state_dim))
dummy_y = np.zeros((1, action_dim))
q_net.partial_fit(dummy_X, dummy_y)

# Create Target Network via deepcopy
target_net = copy.deepcopy(q_net)


# 2. Replay Buffer
class ReplayBuffer:

    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )

    def __len__(self):
        return len(self.buffer)


replay_buffer = ReplayBuffer(capacity=50000)

# Hyperparameters
BATCH_SIZE = 64
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 0.995
TARGET_UPDATE_FREQ = 250  # Steps between target network syncs
episodes = 800

epsilon = EPSILON_START
total_steps = 0
recent_rewards = []

print("Training DQN with Scikit-Learn MLPRegressor...")

for episode in range(1, episodes + 1):
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        total_steps += 1

        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            # Predict Q-values for current state
            q_values = q_net.predict(state.reshape(1, -1))[0]
            action = np.argmax(q_values)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Store transition (use terminated for true environment ends)
        replay_buffer.push(
            state, action, reward, next_state, float(terminated)
        )
        state = next_state
        total_reward += reward

        # Train Network
        if len(replay_buffer) >= BATCH_SIZE:
            states, actions, rewards, next_states, dones = (
                replay_buffer.sample(BATCH_SIZE)
            )

            # Predict current Q-values for all actions
            target_y = q_net.predict(states)

            # Predict next-state Q-values using Target Network
            next_q_values = target_net.predict(next_states)
            max_next_q = np.max(next_q_values, axis=1)

            # Bellman Update: update ONLY the target Q-value for the action taken
            for i in range(BATCH_SIZE):
                a = actions[i]
                r = rewards[i]
                d = dones[i]
                if d:
                    target_y[i, a] = r
                else:
                    target_y[i, a] = r + GAMMA * max_next_q[i]

            # Perform incremental gradient descent step
            q_net.partial_fit(states, target_y)

        # Sync Target Network periodically
        if total_steps % TARGET_UPDATE_FREQ == 0:
            target_net = copy.deepcopy(q_net)

    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
    recent_rewards.append(total_reward)
    avg_reward = np.mean(recent_rewards[-50:])

    print(
        f"Episode {episode:4d} | Reward: {total_reward:6.1f} | Avg (50): {avg_reward:6.1f} | Epsilon: {epsilon:.2f}",
        end="\r",
        flush=True,
    )

    if episode % 50 == 0:
        print(
            f"\nEpisode {episode:4d} | Average Reward (last 50): {avg_reward:6.1f}"
        )

env.close()