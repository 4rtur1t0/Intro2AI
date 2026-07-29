from qlearning.qlearning import QLearning
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

def train_qlearning():
    #environment = gym.make('CliffWalking-v1', render_mode='human')
    environment = gym.make('CliffWalking-v1')
    qlearning = QLearning(environment=environment)
    print('LET US LEARN NOW!')
    qlearning.train(total_episodes_train=100)
    qlearning.save_q_table(filename='qtable.npy')

    print('Tabla Q:')
    print(qlearning.q_table)
    plt.plot(qlearning.results.total_rewards_episode)
    plt.show()
    counts, bins = np.histogram(qlearning.results.total_rewards_episode, bins=1000)
    plt.stairs(counts, bins)
    plt.show()





if __name__ == "__main__":
    train_qlearning()
