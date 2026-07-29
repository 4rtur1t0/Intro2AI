import numpy as np
from qlearning.qlearning import QLearning
import gymnasium as gym
import matplotlib.pyplot as plt

def train_qlearning():
    # use render_mode="human" para observar el entorno gráficamente
    environment = gym.make('Taxi-v3')
    qlearning = QLearning(environment=environment)
    alphas = np.linspace(0, 1.0, 3)  # alpha
    gammas = np.linspace(0, 1.0, 3)  # gamma
    print('LET US LEARN NOW!')
    total_results = []
    for alpha in alphas:
        for gamma in gammas:
            print('Traininig with alpha: {}, gamma: {}'.format(alpha, gamma))
            qlearning.learning_rate = alpha
            qlearning.discount_rate = gamma
            qlearning.train(total_episodes=500)
            qlearning.save_q_table(filename='qtable_taxi.npy')
            total_results.append(qlearning.results)
    total_results = np.array(total_results)

    plt.figure()
    for i in range(len(alphas)):
        for j in range(len(gammas)):
            k = i*len(alphas) + j
            plt.plot(range(len(total_results[k])), total_results[k])
            plt.legend(['Alpha: {}, Gamma: {}'.format(alphas[i], gammas[j])])
    plt.title('Sum of rewards at test episodes')
    plt.show()


if __name__ == "__main__":
    train_qlearning()
