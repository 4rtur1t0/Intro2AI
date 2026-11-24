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
# import json
# import os
# import datetime
from libAI.epsilon_greedy import EpsilonGreedyGeom
# from libAI.results import Results
from tqdm import tqdm

class QLearningDQN():
    def __init__(self, environment, params={}):
        self.env = environment
        # Hyperparameters
        self.batch_size = params.get('batch_size', 64)
        self.gamma = params.get('gamma', 0.99)
        self.epsilon_max = params.get('epsilon_max', 1.0)
        self.epsilon_min = params.get('epsilon_min', 0.01)
        #self.epsilon_step = None  # computed later
        self.epsilon_percentage = params.get('epsilon_percentage', 0.25) # ratio de episodios para llegar a epsilon_min
        # Each 50 episodes of training, test 10 times without updating your knowledge
        self.training_tests = params.get('training_tests', (50, 50))
        #self.epsilon = self.epsilon_max
        self.target_update_freq = params.get('target_update_freq', 250)  # Steps between target network syncs
        self.hidden_layer_sizes = params.get('hidden_layer_sizes', (64, 64))
        # Initialize Neural Networks. Se usa un MLPRgressor de Scikit-Learn para aproximar Q
        self.q_net = MLPRegressor(hidden_layer_sizes=self.hidden_layer_sizes,
                                  activation="relu",
                                  solver="adam",
                                  learning_rate_init=0.0005,
                                  max_iter=1)
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n
        # Run a single dummy partial_fit to initialize network weights and dimensions
        dummy_X = np.zeros((1, state_dim))
        dummy_y = np.zeros((1, action_dim))
        self.q_net.partial_fit(dummy_X, dummy_y)
        # Create Target Network via deepcopy
        self.target_net = copy.deepcopy(self.q_net)
        # El buffer para guardar el mini-batch
        self.replay_buffer = ReplayBuffer(capacity=params.get('replay_buffer_size', 50000))

    def train(self, total_episodes):
        epsilon_greedy = EpsilonGreedyGeom(total_episodes=total_episodes,
                                           epsilon_max=self.epsilon_max,
                                           epsilon_min=self.epsilon_min,
                                           percentage_target=self.epsilon_percentage)
        epsilons = []
        rewards_train = []
        rewards_inline_test = []
        avg_window = 500
        total_steps = 0
        print("Training DQN with Scikit-Learn MLPRegressor...")
        pbar = tqdm(range(total_episodes), desc='DQN-learning train')
        for episode in pbar:
            state, _ = self.env.reset()
            total_reward = 0
            done = False
            while not done:
                total_steps += 1
                action = self.exploration_exploitation(state, epsilon_greedy)
                # apply action on environment and agent
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                # Store transition (use terminated for true environment ends)
                self.replay_buffer.push(state, action,
                                        reward, next_state,
                                        float(terminated))
                state = next_state
                total_reward += reward
                # IMPORTANTE: se entrena la red neuronal online en la train_networks
                self.train_networks(total_steps=total_steps)
            rewards_train.append(total_reward)
            epsilons.append(epsilon_greedy.epsilon)
            if episode % self.training_tests[0] == 0:
                rewards_test = self.test(total_episodes=self.training_tests[1])
                rewards_inline_test.append(np.mean(rewards_test))
            # Afegeix informació a la barra
            pbar.set_postfix(mean_train_reward=np.mean(rewards_train[-avg_window:]),
                             epsilon=epsilon_greedy.epsilon,
                             mean_test_reward=np.mean(rewards_test))
            # Reduce epsilon (less exploration, more exploitation as time goes on)
            epsilon_greedy.step()
        return rewards_train, rewards_inline_test, epsilons

    def train_networks(self, total_steps):
        """
        Train the online network Q (online, q_net)
        Use target_network for stability to generate the targets_y.
        :param total_steps:
        :return:
        """
        if len(self.replay_buffer) >= self.batch_size:
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
            # Predict current Q-values for all actions
            target_y = self.q_net.predict(states)
            # Predict next-state Q-values using Target Network
            # target_net es Q^
            next_q_values = self.target_net.predict(next_states)
            max_next_q = np.max(next_q_values, axis=1)
            # Bellman Update: se actualiza los Q-values solamente para la acción a que se realizó
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
            # SOLAMENTE un paso de actualización por gradient descent
            self.q_net.partial_fit(states, target_y)
        # Sync Target Network periodically
        if total_steps % self.target_update_freq == 0:
            self.target_net = copy.deepcopy(self.q_net)

    def exploration_exploitation(self, state, epsilon_greedy):
        # Epsilon-greedy action selection
        if epsilon_greedy.random_action():
            action = self.env.action_space.sample()
        else:
            # Se usa Q-online para hallar Q(s, a)... que tiene como salida 4 valores
            q_values = self.q_net.predict(state.reshape(1, -1))[0]
            # se halla el máximo de los 4 valores aproximados
            action = np.argmax(q_values)
        return action

    def test(self, total_episodes):
        results_test = []
        # for episode in range(total_episodes):
        pbar = tqdm(range(total_episodes), desc='DQN-learning test')
        for episode in pbar:
            state, info = self.env.reset()
            total_reward = 0
            while True:
                # Greedy action selection using the q_net table
                q_values = self.q_net.predict(state.reshape(1, -1))#[0]
                # se halla el máximo de los 4 valores aproximados
                action = np.argmax(q_values)
                # Take action, observe new state and reward
                next_state, reward, terminated, truncated, info = self.env.step(action)
                total_reward += reward
                if self.env.render_mode == 'human':
                    time.sleep(.05)
                if terminated or truncated:
                    break
                # Move to the next state
                state = next_state
            results_test.append(total_reward)
        return results_test

    def read_model(self, filename):
        with open(filename, "rb") as f:
            self.q_net = pickle.load(f)
        self.target_net = copy.deepcopy(self.q_net)

    def save_model(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.q_net, f)


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
