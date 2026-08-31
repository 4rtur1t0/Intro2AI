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
import datetime
from libAI.epsilon_greedy import EpsilonGreedyGeom


class QLearningDQN():
    def __init__(self, environment, params={}):
        self.env = environment
        # Hyperparameters
        self.batch_size = params.get('batch_size', 64)
        self.gamma = params.get('gamma', 1.0)
        self.epsilon_max = params.get('epsilon_max', 1.0)
        self.epsilon_min = params.get('epsilon_min', 0.01)
        #self.epsilon_step = None  # computed later
        self.epsilon_percentage = params.get('epsilon_percentage', 0.25) # ratio de episodios para llegar a epsilon_min
        # Each 50 episodes of training, test 10 times without updating your knowledge
        self.training_tests = params.get('training_tests', (50, 10))
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
        # para guardar resultados
        self.results = []
        self.avg_window = 100

    def train(self, total_episodes):
        epsilon_greedy = EpsilonGreedyGeom(total_episodes=total_episodes,
                                           epsilon_max=self.epsilon_max,
                                           epsilon_min=self.epsilon_min,
                                           percentage_target=self.epsilon_percentage)
        total_steps = 0
        recent_rewards = []
        self.results = []
        print("Training DQN with Scikit-Learn MLPRegressor...")
        for episode in range(total_episodes):
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
            # reducimos epsilon
            epsilon_greedy.step()
            self.inline_test(episode, epsilon_greedy)
            recent_rewards.append(total_reward)
            avg_reward = np.mean(recent_rewards[-self.avg_window:])
            # self.results.append([total_reward, avg_reward, 100*epsilon_greedy.epsilon])
            print(f"TRAIN Episode {episode:4d} | Total reward: {total_reward:6.1f} | Avg (100): {avg_reward:6.1f} | Epsilon: {epsilon_greedy.epsilon:.2f}",
                   end="\r", flush=True)
            if episode % 1 == 0:
                 print()
        #self.env.close()
        return self.q_net

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

    def test(self, total_episodes, save_results=False):
        recent_rewards = []
        for episode in range(total_episodes):
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
            recent_rewards.append(total_reward)
            #avg_reward = np.mean(self.recent_rewards[-self.avg_window:])
            if save_results:
                self.results.append([episode, total_reward, 0])
            print(f"Episode {episode:4d} | Total reward: {total_reward:6.1f}", end='\r')
            if episode % 10 == 0:
                print()
        return recent_rewards

    def inline_test(self, episode, epsilon_greedy):
        # inline test
        if episode % self.training_tests[0] == 0:
            test_train_results = self.test(total_episodes=self.training_tests[1])
            test_train_results = np.mean(test_train_results)
            self.results.append([episode, test_train_results, 100 * epsilon_greedy.epsilon])
            print(f"INLINE TEST {episode:4d} | Avg reward ({self.training_tests[1]}): {test_train_results:6.1f} | Epsilon: 0.")

    def read_model(self, filename):
        with open(filename, "rb") as f:
            self.q_net = pickle.load(f)
        self.target_net = copy.deepcopy(self.q_net)

    def save_model(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.q_net, f)

    def save_results(self, experiment_name=None, type_result='train'):
        """
        Save results as json
        :param experiment_name:
        :param type_result:
        :return:
        """
        now = datetime.datetime.now()
        time_string = now.strftime("%Y%m%d_%H%M%S")
        experiment_filename = f"results/{type_result}/{time_string}.json"
        experiment_data = {
            'experiment_name': experiment_name,
            'time_string': time_string,
            "experiment_filename": experiment_filename,
            "params": {'gamma': self.gamma,
                       'hidden_layers': self.hidden_layer_sizes,
                       'epsilon_max': self.epsilon_max,
                       'epsilon_min': self.epsilon_min,
                       'epsilon_percentage': self.epsilon_percentage},
            "episodes": [r[0] for r in self.results],
            "rewards": [r[1] for r in self.results],
            #"avg_rewards": [r[1] for r in self.results],
            "100*epsilon": [r[2] for r in self.results],
            "global_mean_reward": np.mean([r[1] for r in self.results]),
            "global_mean_variance": np.std([r[1] for r in self.results])
        }
        # Ensure directory exists before writing
        os.makedirs(f"results/{type_result}", exist_ok=True)
        with open(experiment_filename, "w") as f:
            json.dump(experiment_data, f)

    def reset_results(self):
        self.results = []


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
