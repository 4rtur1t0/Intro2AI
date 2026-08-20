"""
    A solution for QLearning.
    Es poden resoldre diferents escenes d'aprenentatge amb aquest mateix disseny.
"""
import numpy as np
import random
import time
from tqdm import tqdm

from libAI.epsilon_greedy import EpsilonGreedy


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
        self.learning_rate = 1.0  # alpha
        self.discount_rate = 1.0  # gamma
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_max = 1.0
        self.epsilon_min = 0.01
        self.percentage_target = 0.25  # Exponential decay rate for exploration
        # para guardar resultados
        self.results = []
        # during training, test each testing_episodes
        self.testing_episodes = 10

    def train(self, total_episodes):
        self.results = []
        print('Training started!', flush=True)
        pbar = tqdm(total=total_episodes, desc= 'Episodios', colour='green')
        epsilon_greedy = EpsilonGreedy(epsilon_max=self.epsilon_max,
                                       epsilon_min=self.epsilon_min,
                                       total_episodes=total_episodes,
                                       percentage_target=self.percentage_target)
        recent_rewards = []
        # Training loop
        for episode in range(total_episodes):
            # print("Episode: ", episode)
            state, info = self.env.reset()
            # sum of rewards for each episode
            total_reward = 0
            if episode % self.testing_episodes == 0:
                random_action = False
            else:
                random_action = True
            while True:
                # Epsilon-greedy action selection
                if  epsilon_greedy.random_action():
                    action = self.env.action_space.sample()  # Explore
                else:
                    action = np.argmax(self.q_table[state])  # Exploit
                # Take action, observe new state and reward
                next_state, reward, terminated, truncated, info = self.env.step(action)
                total_reward += reward
                # Actualiza la tabla Q
                self.update_q_table(state, action, next_state, reward)
                if terminated or truncated:
                    #if not random_action:
                    #    self.results.append(srt)
                    break
                # Move to next state
                state = next_state
            pbar.update(1)
            # Reduce epsilon
            epsilon_greedy.step()
            # test online en algunos casos
            self.inline_test(episode, epsilon_greedy)
            recent_rewards.append(total_reward)
            avg_reward = np.mean(recent_rewards[-self.avg_window:])
            print(f"TRAIN Episode {episode:4d} | Total reward: {total_reward:6.1f} | Avg (100): {avg_reward:6.1f} | Epsilon: {epsilon_greedy.epsilon:.2f}",
                end="\r", flush=True)
            if episode % 20 == 0:
                print()
        print()
        print("Training finished! Your Q-table is optimized.")
        self.env.close()
        return self.q_table


    def update_q_table(self, state, action, next_state, reward):
        # Update Q-table using the Bellman Equation
         self.q_table[state, action] = (self.q_table[state, action] +
                                        self.learning_rate * (
                                                    reward + self.discount_rate * np.max(self.q_table[next_state]) -
                                                    self.q_table[state, action]))

    def test(self, total_episodes):
        print('Test started')
        # test loop
        for episode in range(total_episodes):
            print("Episode: ", episode)
            state, info = self.env.reset()
            srt = 0
            while True:
                # Greedy action selection
                action = np.argmax(self.q_table[state])  # Exploit
                # Take action, observe new state and reward
                next_state, reward, terminated, truncated, info = self.env.step(action)
                srt += reward
                time.sleep(.1)
                if terminated or truncated:
                    self.results.append(srt)
                    # self.results.save_data(sum_rewards_per_episodei=srt)
                    break
                # Move to the next state
                state = next_state
        print("Test finished!")
        self.env.close()

    def inline_test(self, episode, epsilon_greedy):
        # inline test
        if episode % self.training_tests[0] == 0:
            test_train_results = self.test(total_episodes=self.training_tests[1])
            test_train_results = np.mean(test_train_results)
            self.results.append([episode, test_train_results, 100 * epsilon_greedy.epsilon])
            print(f"INLINE TEST {episode:4d} | Avg reward ({self.training_tests[1]}): {test_train_results:6.1f} | Epsilon: 0.")


    def create_random_q_table(self):
        self.q_table = np.random.rand(self.state_size, self.action_size)

    def read_q_table(self, filename):
        with open(filename, 'rb') as f:
            self.q_table = np.load(f)

    def save_q_table(self, filename):
        with open(filename, 'wb') as f:
            np.save(f, self.q_table)
