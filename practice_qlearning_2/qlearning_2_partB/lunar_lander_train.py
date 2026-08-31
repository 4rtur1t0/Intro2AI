from qlearning_DQN.qlearning_DQN import QLearningDQN
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np


def plot_results(qlearning):
    results = np.array(qlearning.results)
    plt.figure()
    #plt.plot(range(len(results)), , label="total reward")
    plt.plot(results[:,0], results[:,1], label="Total reward (inline test average)")
    plt.plot(results[:,0], results[:,2], label="100*epsilon")
    plt.xlabel("Episode")
    plt.ylabel("Reward/epsilon")
    plt.legend()
    plt.show()

def train_deep_qlearning():
    try:
        environment = gym.make("LunarLander-v3")
    except Exception:
        environment = gym.make("LunarLander-v2")
    total_episodes_train = 10000
    params = {'gamma': 0.99,
              'epsilon_percentage': 0.25,
              'hidden_layer_sizes': (64, 64),
              'training_tests': (50, 50)}
    qlearning = QLearningDQN(environment=environment, params=params)
    # podemos leer el último modelo guardado para seguir entrenando sobre él
    #qlearning.read_model(filename='q_learning_DQN_lunar_lander.pkl')
    qlearning.train(total_episodes=total_episodes_train)
    qlearning.save_model(filename='q_learning_DQN_lunar_lander.pkl')
    qlearning.save_results(experiment_name=str(params), type_result='train')
    plot_results(qlearning)

if __name__ == "__main__":
    train_deep_qlearning()
