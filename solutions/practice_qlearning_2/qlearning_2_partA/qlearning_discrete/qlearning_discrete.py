"""
    A solution for QLearning.
    Es tracta de tabular Q learning on es discretitza un estat continu.
    Es poden resoldre diferents escenes d'aprenentatge amb aquest mateix disseny.
"""
import numpy as np
import random
import time
from collections import defaultdict
import pickle
import os
import json

# Non-uniform bin boundaries: tighter resolution around 0 for fine control
X_BINS = np.array([-0.4, -0.15, -0.05, 0.05, 0.15, 0.4])
Y_BINS = np.array([0.05, 0.2, 0.5, 0.9])
VX_BINS = np.array([-0.5, -0.2, -0.05, 0.05, 0.2, 0.5])
VY_BINS = np.array([-0.8, -0.3, -0.1, 0.0, 0.2])
ANGLE_BINS = np.array([-0.4, -0.1, -0.02, 0.02, 0.1, 0.4])
ANG_VEL_BINS = np.array([-0.6, -0.2, 0.2, 0.6])


class QLearningD():
    """
    Q learning con discretización
    """
    def __init__(self, environment):
        self.env = environment
        self.action_size = self.env.action_space.n
        # importante, la tabla Q la constituye un diccionario.
        # Devuelve ceros si se solicita una clave que no existe.
        self.q_table = defaultdict(lambda: np.zeros(self.action_size))
        # Hyperparameters
        self.alpha = 0.2  # alpha
        self.gamma = 0.99  # gamma
        self.epsilon = 1.0  # Exploration rate
        self.max_epsilon = 1.0
        self.min_epsilon = 0.01
        #self.epsilon_decay = 0.99999  # Exponential decay rate for exploration
        self.avg_window = 100 # una ventana para hacer la media móvil del resultado
        # para guardar resultados
        self.results = []

    def train(self, total_episodes):
        decay_episodes = int(total_episodes*0.8) # leave 20 % at min epsilon
        epsilon_step = (self.max_epsilon - self.min_epsilon) / decay_episodes
        self.results = []
        recent_rewards = []
        print("Training started!\n\n")
        # Training loop
        for episode in range(total_episodes):
            state, info = self.env.reset()
            state_d = self.discretize_state(state)
            # sum of rewards for each episode
            total_reward = 0
            while True:
                # Epsilon-greedy action selection
                if random.uniform(0, 1) < self.epsilon:
                    action = self.env.action_space.sample()  # Explore
                else:
                    action = np.argmax(self.q_table[state_d])  # Exploit
                # Take action, observe new state and reward
                next_state, reward, terminated, truncated, info = self.env.step(action)
                next_state_d = self.discretize_state(next_state)
                total_reward += reward
                # Actualiza la tabla
                done = terminated or truncated
                self.update_q_table(state_d, action, next_state_d, reward, done)
                if done:
                    break
                # Move to next state
                state_d = next_state_d
            # Reduce epsilon (less exploration, more exploitation as time goes on)
            #self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            self.epsilon -= epsilon_step
            self.epsilon = max(self.min_epsilon, self.epsilon)
            recent_rewards.append(total_reward)
            avg_reward = np.mean(recent_rewards[-self.avg_window:])
            self.results.append([total_reward, avg_reward, 100 * self.epsilon, len(self.q_table)/1000])
            print(f"Episode {episode:4d} | Reward: {total_reward:6.1f} | Avg (last 100): {avg_reward:6.1f} | Epsilon: {self.epsilon:.2f}",
                  end="\r", flush=True)
            if episode % self.avg_window == 0:
                print()
            self.env.close()
        return self.q_table

    def update_q_table(self, state_d, action, next_state_d, reward, done):
        if not done:
            # Update Q-table using the Bellman Equation
            self.q_table[state_d][action] = self.q_table[state_d][action] + \
                                            self.alpha * (reward
                                                          + self.gamma * np.max(self.q_table[next_state_d]) -
                                                            self.q_table[state_d][action])
        else:
            # Update Q-table using the Bellman Equation, with Q(s',a)=0
            self.q_table[state_d][action] = (self.q_table[state_d][action] +\
                                             self.alpha * (reward - self.q_table[state_d][action]))

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

    def save_results(self):
        experiment_filename = (f"results/gamma_{self.gamma:.2f}_alpha_{self.alpha:.2f}.json")
        experiment_data = {
            "experiment_filename": experiment_filename,
            "params": {"gamma": self.gamma, "alpha": self.alpha},
            "episodes": list(range(len(self.results))),
            "rewards": [r[0] for r in self.results],
            "avg_rewards": [r[1] for r in self.results],
            "100*epsilon": [r[2] for r in self.results]
        }
        # Ensure directory exists before writing
        os.makedirs("results", exist_ok=True)
        with open(experiment_filename, "w") as f:
            json.dump(experiment_data, f)

