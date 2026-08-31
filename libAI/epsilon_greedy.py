"""
    A simple class to handle epsilon_greedy exploration/exploitation.
"""
import numpy as np
import random
import matplotlib.pyplot as plt

class EpsilonGreedy():
    """
    Base class
    """
    def __init__(self, epsilon_max, epsilon_min, total_episodes, percentage_target):
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon = epsilon_max
        self.total_episodes = total_episodes
        self.target_episodes = percentage_target*total_episodes
        #self.epsilon_decay = np.power(epsilon_min/epsilon_max, 1.0/self.target_episodes)
        self.epsilon_data = []
        self.epsilon_data.append(self.epsilon)


    def random_action(self):
        if random.uniform(0.0, 1.0) < self.epsilon:
            return True
        return False

    def reset(self):
        self.epsilon = self.epsilon_max

    def plot(self, title):
        plt.figure()
        plt.plot(range(len(self.epsilon_data)), self.epsilon_data)
        plt.title(title)
        plt.xlabel("Episodes")
        plt.ylabel("Epsilon")
        plt.show()

class EpsilonGreedyGeom(EpsilonGreedy):
    def __init__(self, epsilon_max, epsilon_min, total_episodes, percentage_target):
        EpsilonGreedy.__init__(self, epsilon_max, epsilon_min, total_episodes, percentage_target)
        self.epsilon_decay = np.power(epsilon_min/epsilon_max, 1.0/self.target_episodes)

    def step(self):
        self.epsilon = max(self.epsilon_min, self.epsilon*self.epsilon_decay)
        self.epsilon_data.append(self.epsilon)

class EpsilonGreedyExp(EpsilonGreedy):
    def __init__(self, epsilon_max, epsilon_min, total_episodes, percentage_target):
        EpsilonGreedy.__init__(self, epsilon_max, epsilon_min, total_episodes, percentage_target)
        self.epsilon_decay = -np.log(epsilon_min/epsilon_max)/self.target_episodes

    def step(self):
        self.epsilon = max(self.epsilon_min, self.epsilon*np.exp(-self.epsilon_decay))
        self.epsilon_data.append(self.epsilon)

class EpsilonGreedyLinear(EpsilonGreedy):
    def __init__(self, epsilon_max, epsilon_min, total_episodes, percentage_target):
        EpsilonGreedy.__init__(self, epsilon_max, epsilon_min, total_episodes, percentage_target)
        self.epsilon_decay = (epsilon_max - epsilon_min)/self.target_episodes

    def step(self):
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)
        self.epsilon_data.append(self.epsilon)


class EpsilonGreedyInverseTime(EpsilonGreedy):
    def __init__(self, epsilon_max, epsilon_min, total_episodes, percentage_target):
        EpsilonGreedy.__init__(self, epsilon_max, epsilon_min, total_episodes, percentage_target)
        self.epsilon_decay = (epsilon_max/epsilon_min-1) / self.target_episodes
        # normalize
        self.epsilon_decay = self.epsilon_decay/epsilon_max

    def step(self):
        self.epsilon = max(self.epsilon_min, self.epsilon/(1+self.epsilon_decay*self.epsilon))
        self.epsilon_data.append(self.epsilon)



