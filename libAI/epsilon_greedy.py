"""
    A simple class to handle epsilon_greedy exploration/exploitation.
"""
import numpy as np
import random



class EpsilonGreedy():
    def __init__(self, epsilon_max, epsilon_min, total_episodes, percentage_target):
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon = epsilon_max
        self.total_episodes = total_episodes
        self.target_episodes = percentage_target*total_episodes
        self.epsilon_decay = np.power(epsilon_min/epsilon_max, 1.0/self.target_episodes)

    def step(self):
        self.epsilon = max(self.epsilon_min, self.epsilon*self.epsilon_decay)

    def random_action(self):
        if random.uniform(0, 1) < self.epsilon:
            return True
        return False

    def reset(self):
        self.epsilon = self.epsilon_max