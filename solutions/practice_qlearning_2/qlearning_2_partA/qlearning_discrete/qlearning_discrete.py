"""
    A solution for QLearning.
    Es tracta de tabular Q learning on es discretitza un estat continu.
    Es poden resoldre diferents escenes d'aprenentatge amb aquest mateix disseny.
"""
import numpy as np
import random
import time
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle

# Non-uniform bin boundaries: tighter resolution around 0 for fine control
X_BINS = np.array([-0.4, -0.15, -0.05, 0.05, 0.15, 0.4])
Y_BINS = np.array([0.05, 0.2, 0.5, 0.9])
VX_BINS = np.array([-0.5, -0.2, -0.05, 0.05, 0.2, 0.5])
VY_BINS = np.array([-0.8, -0.3, -0.1, 0.0, 0.2])
ANGLE_BINS = np.array([-0.4, -0.1, -0.02, 0.02, 0.1, 0.4])
ANG_VEL_BINS = np.array([-0.6, -0.2, 0.2, 0.6])


class QLearningD():
    def __init__(self, environment):
        self.env = environment
        # Initialize Q-table with zeros (500 states x 6 actions)
        #self.state_size = self.env.observation_space.n
        self.action_size = self.env.action_space.n
        # importante, la tabla Q se inicializa a ceros por defecto
        # se puede cargar desde un fichero con el método load_q_table
        # se puede guardar con el método save_q_table
        #self.q_table = np.zeros((self.state_size, self.action_size))
        self.q_table = defaultdict(lambda: np.zeros(self.action_size))

        # Hyperparameters
        self.learning_rate = 0.3  # alpha
        self.discount_rate = 1.0  # gamma
        self.epsilon = 1.0  # Exploration rate
        self.max_epsilon = 1.0
        self.min_epsilon = 0.01
        self.decay_rate = 0.9996  # Exponential decay rate for exploration

        # run test episodes each 1000 episodes
        self.test_episodes_each = 1000
        # para guardar resultados
        self.results = []

    def train(self, total_episodes):
        self.results = []
        print("Training started!\n\n")
        # Training loop
        for episode in range(total_episodes):
            #print("Episode: ", episode)
            print(f"Episodio actual: {episode}", end="\r", flush=True)
            state, info = self.env.reset()
            state_d = self.discretize_state(state)
            # sum of rewards for each episode
            srt = 0
            # if random_action=False, then only the q_table is used
            # throughout the whole episode
            if episode % self.test_episodes_each == 0:
                random_action = False
            else:
                random_action = True
            while True:
                # Epsilon-greedy action selection
                if random_action and (random.uniform(0, 1) < self.epsilon):
                    action = self.env.action_space.sample()  # Explore
                else:
                    action = np.argmax(self.q_table[state_d])  # Exploit
                # Take action, observe new state and reward
                next_state, reward, terminated, truncated, info = self.env.step(action)
                next_state_d = self.discretize_state(next_state)
                srt += reward
                if terminated or truncated:
                    if not random_action:
                        print('Total reward:', srt)
                        print('Total visited states: ', len(self.q_table))
                        self.results.append(srt)
                    break
                # Actualiza la tabla
                self.update_q_table(state_d, action, next_state_d, reward)
                # Move to next state
                state_d = next_state_d
            # Reduce epsilon (less exploration, more exploitation as time goes on)
            self.epsilon *= self.decay_rate
            self.epsilon = max(self.epsilon, self.min_epsilon)
        print("Training finished! Your Q-table is optimized.")
        self.env.close()
        return self.q_table

    def update_q_table(self, state_d, action, next_state_d, reward):
        # Update Q-table using the Bellman Equation
         self.q_table[state_d][action] = (self.q_table[state_d][action] +
                                        self.learning_rate * (
                                                    reward + self.discount_rate * np.max(self.q_table[next_state_d]) -
                                                    self.q_table[state_d][action]))

    def test(self, total_episodes):
        print('Test started')
        # test loop
        for episode in range(total_episodes):
            print("Episode: ", episode)
            state, info = self.env.reset()
            state_d = self.discretize_state(state)
            srt = 0
            while True:
                # Greedy action selection
                action = np.argmax(self.q_table[state_d])  # Exploit
                # Take action, observe new state and reward
                next_state, reward, terminated, truncated, info = self.env.step(action)
                next_state_d = self.discretize_state(next_state)
                srt += reward
                time.sleep(.1)
                if terminated or truncated:
                    self.results.append(srt)
                    print('Total reward of episode:', srt)
                    break
                # Move to the next state
                state_d = next_state_d
        print("Test finished!")
        self.env.close()

    def create_random_q_table(self):
        # Distribución normal centrada en 0 (media=0, std=1)
        self.q_table = defaultdict(lambda: np.random.randn(self.action_size))
    #    self.q_table = np.random.rand(self.state_size, self.action_size)

    def read_q_table(self, filename):
        with open(filename, "rb") as f:
            raw_q_table = pickle.load(f)
        self.q_table = defaultdict(lambda: np.zeros(self.action_size), raw_q_table)

    def save_q_table(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(dict(self.q_table), f)

    def discretize_state(self, state):
        """Maps the 8D continuous state into a discrete tuple of bin indices."""
        x, y, vx, vy, angle, ang_vel, leg1, leg2 = state
        return (
            int(np.digitize(x, X_BINS)),
            int(np.digitize(y, Y_BINS)),
            int(np.digitize(vx, VX_BINS)),
            int(np.digitize(vy, VY_BINS)),
            int(np.digitize(angle, ANGLE_BINS)),
            int(np.digitize(ang_vel, ANG_VEL_BINS)),
            int(leg1),
            int(leg2),
        )

