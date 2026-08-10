"""
    A solution for Deep QLearning.
    Q learning using a neural network to approximate Q
"""
import numpy as np
import random
import time
import pickle
from collections import deque
from sklearn.neural_network import MLPRegressor
import copy
import json
import os


class QLearningDQN():
    def __init__(self, environment):
        self.env = environment
        # Hyperparameters
        self.batch_size = 64
        #self.gamma = 0.99
        self.gamma = 1.0
        self.epsilon_start = 1.0
        self.epsilon_end = 0.02
        #self.epsilon_decay = 0.995
        self.epsilon_decay = 0.98
        self.target_update_freq = 250  # Steps between target network syncs
        self.epsilon = self.epsilon_start
        self.hidden_layer_sizes = (64, 64)
        # Initialize Neural Networks. Se usa un MLPRgressor de Scikit-Learn para aproximar Q
        self.q_net = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation="relu",
            solver="adam",
            learning_rate_init=0.0005,
            max_iter=1,
        )
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n
        # Run a single dummy partial_fit to initialize network weights and dimensions
        dummy_X = np.zeros((1, state_dim))
        dummy_y = np.zeros((1, action_dim))
        self.q_net.partial_fit(dummy_X, dummy_y)
        # Create Target Network via deepcopy
        self.target_net = copy.deepcopy(self.q_net)
        # El buffer para guardar el mini-batch
        self.replay_buffer = ReplayBuffer(capacity=50000)

        # run test episodes each 1000 episodes
        #self.test_episodes_each = 1000
        # para guardar resultados
        self.results = []

    def train(self, total_episodes):
        total_steps = 0
        recent_rewards = []
        print("Training DQN with Scikit-Learn MLPRegressor...")
        for episode in range(0, total_episodes):
            state, _ = self.env.reset()
            total_reward = 0
            done = False
            while not done:
                total_steps += 1
                # Epsilon-greedy action selection
                if random.uniform(0, 1) < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    # Predict Q-values for current state. Se usa Q-online para
                    # hallar Q(s, a)... que tiene como salida 4 valores
                    q_values = self.q_net.predict(state.reshape(1, -1))[0]
                    # se halla el máximo de los 4 valores aproximados
                    action = np.argmax(q_values)
                # apply action on environment and agent
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                # Store transition (use terminated for true environment ends)
                self.replay_buffer.push(state, action,
                                        reward, next_state,
                                        float(terminated))
                state = next_state
                total_reward += reward
                # IMPORTANTE: la actualización de la tabla Q se cambia
                # por la estimación de la red neuronal que la aproxima
                # en esta función se agrupan todas las operaciones sobre las redes neuronales
                self.train_networks(total_steps=total_steps)
            # reducimos epsilon, calculamos la evolución
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            recent_rewards.append(total_reward)
            avg_reward = np.mean(recent_rewards[-50:])
            self.results.append([total_reward, avg_reward, 100*self.epsilon])
            print(f"Episode {episode:4d} | Reward: {total_reward:6.1f} | Avg (50): {avg_reward:6.1f} | Epsilon: {self.epsilon:.2f}",
                end="\r",
                flush=True)
            if episode % 50 == 0:
                print(f"\nEpisode {episode:4d} | Average Reward (last 50): {avg_reward:6.1f}")
        self.env.close()
        return self.q_net

    def train_networks(self, total_steps):
        # Train Network
        if len(self.replay_buffer) >= self.batch_size:
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
            # Predict current Q-values for all actions
            target_y = self.q_net.predict(states)
            # Predict next-state Q-values using Target Network
            # Esta es Q^
            next_q_values = self.target_net.predict(next_states)
            max_next_q = np.max(next_q_values, axis=1)
            # Bellman Update: update ONLY the target Q-value for the action taken
            # IMPORTANTE: la red neuronal aproxima la salida Q(s, a) para cada una de las
            # 4 acciones posibles. En este bucle, calculamos los pares (X, y) para
            # actualizar q_net
            for i in range(self.batch_size):
                a = actions[i]
                r = rewards[i]
                d = dones[i]
                if d:
                    target_y[i, a] = r
                else:
                    target_y[i, a] = r + self.gamma * max_next_q[i]
            # Perform an incremental gradient descent step
            # SOLAMENTE un paso de actualización
            self.q_net.partial_fit(states, target_y)
        # Sync Target Network periodically
        if total_steps % self.target_update_freq == 0:
            self.target_net = copy.deepcopy(self.q_net)

    def test(self, total_episodes):
        print('Test started')
        # test loop
        for episode in range(total_episodes):
            print("Episode: ", episode)
            state, info = self.env.reset()
            total_reward = 0
            while True:
                # Greedy action selection using the q_net table
                q_values = self.q_net.predict(state.reshape(1, -1))[0]
                # se halla el máximo de los 4 valores aproximados
                action = np.argmax(q_values)
                # Take action, observe new state and reward
                next_state, reward, terminated, truncated, info = self.env.step(action)
                total_reward += reward
                time.sleep(.05)
                if terminated or truncated:
                    self.results.append(total_reward)
                    print('Total reward of episode:', total_reward)
                    break
                # Move to the next state
                state = next_state
        print("Test finished!")
        self.env.close()

    def read_model(self, filename):
        with open(filename, "rb") as f:
            self.q_net = pickle.load(f)
        self.target_net = copy.deepcopy(self.q_net)

    def save_model(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.q_net, f)

    def save_results(self):
        experiment_filename = f"results/gamma_{self.gamma:.2f}_eps_dcy_{self.epsilon_decay:.2f}_net_{self.hidden_layer_sizes}.json"
        experiment_data = {
            "experiment_filename": experiment_filename,
            "params": {"gamma": 0.99, "hidden_layers": self.hidden_layer_sizes,
                       "epsilon_decay": self.epsilon_decay},
            "episodes": list(range(len(self.results))),
            "rewards": [r[0] for r in self.results],
            "avg_rewards": [r[1] for r in self.results],
            "100*epsilon": [r[2] for r in self.results]
        }
        # Ensure directory exists before writing
        os.makedirs("results", exist_ok=True)
        with open(experiment_filename, "w") as f:
            json.dump(experiment_data, f)

# Replay Buffer
class ReplayBuffer:

    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def __len__(self):
        return len(self.buffer)
