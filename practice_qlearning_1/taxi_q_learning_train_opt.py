"""
Se desea hacer una imagen con el resultado sobre test al entrenar con diferentes alphas y gammas.
"""
import numpy as np
from qlearning.qlearning import QLearning
import gymnasium as gym
import matplotlib.pyplot as plt

def train_qlearning():
    try:
        environment = gym.make('Taxi-v3')
    except:
        environment = gym.make('Taxi-v4')
    total_episodes_train = 2000
    total_episodes_test = 500
    alphas = np.linspace(0.05, 1.0, 10)  # alpha
    gammas = np.linspace(0.05, 1.0, 10)  # gamma
    print('LET US LEARN NOW!')
    total_results = np.zeros((len(alphas), len(gammas)))
    for i in range(len(alphas)):
        for j in range(len(gammas)):
            # Caution, restart the object so that the Q table and results are reset
            params = {'alpha': alphas[i], 'gamma': gammas[j]}
            qlearning = QLearning(environment=environment, params=params)
            qlearning.train(total_episodes=total_episodes_train)
            results = qlearning.test(total_episodes=total_episodes_test)
            total_results[i][j]=np.mean(results)
    fig, ax = plt.subplots()
    im = ax.imshow(total_results)
    # Show all ticks and label them with the respective list entries
    ax.set_yticks(range(len(alphas)), labels=['{0:.1f}'.format(alpha) for alpha in alphas])
    ax.set_xticks(range(len(gammas)), labels=['{0:.1f}'.format(gamma) for gamma in gammas])
    # Loop over data dimensions and create text annotations.
    for i in range(len(alphas)):
        for j in range(len(gammas)):
            text = ax.text(j, i, '{0:.1f}'.format(total_results[i, j]),
                           ha="center", va="center", color="w")
    ax.set_title("Reward medio en episodios de test intermedios")
    plt.xlabel("gamma")
    plt.ylabel("alfa")
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_qlearning()
