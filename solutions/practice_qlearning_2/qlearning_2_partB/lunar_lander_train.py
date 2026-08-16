from qlearning_DQN.qlearning_DQN import QLearningDQN
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np


def plot_results(qlearning):
    results = np.array(qlearning.results)
    plt.figure()
    plt.plot(range(len(results)), results[:,0], label="total reward")
    plt.plot(range(len(results)), results[:,1], label="total reward (average)")
    plt.plot(range(len(results)), results[:,2], label="100*epsilon")
    plt.legend()
    plt.show()

def train_deep_qlearning():
    try:
        environment = gym.make("LunarLander-v3")
    except Exception:
        environment = gym.make("LunarLander-v2")
    qlearning = QLearningDQN(environment=environment)
    # podemos leer el último modelo guardado para seguir entrenando sobre él
    #qlearning.read_model(filename='mlpregressor_qtable_lunar_lander.pkl')
    qlearning.train(total_episodes=60)
    qlearning.save_model(filename='mlpregressor_qtable_lunar_lander.pkl')
    qlearning.save_results()
    plot_results(qlearning)

if __name__ == "__main__":
    train_deep_qlearning()
