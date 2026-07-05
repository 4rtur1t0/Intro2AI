"""
    A solution for QLearning.
    Es poden resoldre diferents escenes d'aprenentatge amb aquest mateix disseny.
"""
import numpy as np
import random
import time
import matplotlib.pyplot as plt

class QLearning():
    def __init__(self, environment):
        self.env = environment
        # Initialize Q-table with zeros (500 states x 6 actions)
        self.state_size = self.env.observation_space.n
        self.action_size = self.env.action_space.n
        # importante, la tabla Q se inicializa a ceros por defecto
        # se puede cargar desde un fichero con el método load_q_table
        # se puede guardar con el método save_q_table
        self.q_table = np.zeros((self.state_size, self.action_size))

        # Hyperparameters
        self.learning_rate = 0.1  # alpha
        self.discount_rate = 0.6  # gamma
        self.epsilon = 1.0  # Exploration rate
        self.max_epsilon = 1.0
        self.min_epsilon = 0.01
        self.decay_rate = 0.01  # Exponential decay rate for exploration
        # para guardar resultados
        self.results = Results()

    def train(self, total_episodes_train):
        print("Training started!\n\n")
        # Training loop
        for episode in range(total_episodes_train):
            print("Episode: ", episode)
            state, info = self.env.reset()
            while True:
                # Step 1: Epsilon-greedy action selection
                if random.uniform(0, 1) < self.epsilon:
                    action = self.env.action_space.sample()  # Explore
                else:
                    action = np.argmax(self.q_table[state])  # Exploit
                # Step 2: Take action, observe new state and reward
                next_state, reward, terminated, truncated, info = self.env.step(action)
                # print('Action, Reward:', action, reward)
                # time.sleep(5)
                self.results.save_data(episode, next_state, reward)
                if terminated or truncated:
                    self.results.q_table = self.q_table
                    break
                # Step 3: Actualiza la tabla
                # usa una de estas funciones.
                self.update_q_table_bellman(state, action, next_state, reward)

                # Move to next state
                state = next_state
            # Reduce epsilon (less exploration, more exploitation as time goes on)
            self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay_rate * episode)

        print("Training finished! Your Q-table is optimized.")
        self.env.close()
        return self.q_table


    def update_q_table_bellman(self, state, action, next_state, reward):
        # Step 3: Update Q-table using the Bellman Equation
        self.q_table[state, action] = (self.q_table[state, action] +
                                       self.learning_rate * (
                                                   reward + self.discount_rate * np.max(self.q_table[next_state]) -
                                                   self.q_table[state, action]))

    def test(self, total_episodes_test):
        print('Test started')
        # test loop
        for episode in range(total_episodes_test):
            print("Episode: ", episode)
            state, info = self.env.reset()
            while True:
                # Step 1: Greedy action selection
                action = np.argmax(self.q_table[state])  # Exploit
                # Step 2: Take action, observe new state and reward
                next_state, reward, terminated, truncated, info = self.env.step(action)
                self.results.save_data(episode, next_state, reward)
                time.sleep(1)
                if terminated or truncated:
                    break
                # Move to the next state
                state = next_state
        print("Test finished!")
        self.env.close()

    def load_random_q_table(self):
        self.q_table = np.random.rand(self.state_size, self.action_size)

    def load_q_table(self, filename):
        with open(filename, 'rb') as f:
            self.q_table = np.load(f)

    def save_q_table(self, filename):
        with open(filename, 'wb') as f:
            np.save(f, self.q_table)


class Results():
    def __init__(self):
        # store the total reward
        self.data = []
        # store the q_table to be plotted
        self.q_table = []

    def save_data(self, episode, state, reward):
        self.data.append([episode, state, reward])

    def plot_data(self):
        print('Plotting data')
        print('Computing mean reward per epidode')
        self.data = np.array(self.data)
        last_episode = self.data[-1][0]
        sum_rewards_per_episode = []
        for episode in range(last_episode):
            print(f"Episode: {episode}", end="\r", flush=True)
            mascara = (self.data[:, 0] == episode)
            submatrix = self.data[mascara]
            s = np.sum(submatrix[:, 2])
            #s = np.mean(submatrix[:, 2])
            #c = np.cov(submatrix[:, 2])
            sum_rewards_per_episode.append(s)
        plt.plot(range(len(sum_rewards_per_episode)), sum_rewards_per_episode)
        plt.legend(['Sum of rewards at each episode'])
        plt.show()
        plt.title("Sum of rewards for each episode")



