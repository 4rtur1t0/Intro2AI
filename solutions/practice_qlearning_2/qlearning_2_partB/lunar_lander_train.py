from qlearning_DQN.qlearning_DQN import QLearningDQN
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np


def train_deep_qlearning():
    try:
        environment = gym.make("LunarLander-v3")
    except Exception:
        environment = gym.make("LunarLander-v2")
    total_episodes_train = 150
    params = {'gamma': 0.99,
              'epsilon_percentage': 0.25,
              'hidden_layer_sizes': (64, 64),
              'training_tests': (20, 5)}
    qlearning = QLearningDQN(environment=environment, params=params)
    # podemos leer el último modelo guardado para seguir entrenando sobre él
    #qlearning.read_model(filename='q_learning_DQN_lunar_lander.pkl')
    results = qlearning.train(total_episodes=total_episodes_train)
    qlearning.save_model(filename='qlearning_DQN_lunar_lander.pkl')
    results.save(experiment_name=str(params))
    results.plot()

if __name__ == "__main__":
    train_deep_qlearning()
