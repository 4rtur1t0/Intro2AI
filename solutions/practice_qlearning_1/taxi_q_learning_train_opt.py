import numpy as np
from qlearning.qlearning import QLearning
import gymnasium as gym
import matplotlib.pyplot as plt

def train_qlearning():
    # use render_mode="human" para observar el entorno gráficamente
    environment = gym.make('Taxi-v4')
    alphas = np.linspace(0, 1.0, 5)  # alpha
    gammas = np.linspace(0, 1.0, 5)  # gamma
    alphas = [0.1, 1.0]
    gammas = [0.1, 1.0]
    print('LET US LEARN NOW!')
    legends = []
    total_results = []
    for alpha in alphas:
        for gamma in gammas:
            # Caution, restart the object so that the Q table and results are reset
            qlearning = QLearning(environment=environment)
            qlearning.learning_rate = alpha
            qlearning.discount_rate = gamma
            qlearning.train(total_episodes=1500)
            total_results.append(qlearning.results)
            legends.append('($\alpha$, $\gamma$)=({}_{})'.format(alpha, gamma))
    total_results = np.array(total_results)

    plt.figure()
    for k in range(len(alphas)*len(gammas)):
            plt.plot(range(len(total_results[k])), total_results[k], label=legends[k], linewidth=4)
            plt.legend(loc='lower right')
    plt.title('Sum of rewards at test episodes (during training)')
    plt.xlabel('Episodes (1/10)')
    plt.ylabel('Sum of rewards at each episode')
    plt.show()


if __name__ == "__main__":
    train_qlearning()
