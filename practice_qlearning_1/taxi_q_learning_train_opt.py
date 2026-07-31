import numpy as np
from qlearning.qlearning import QLearning
import gymnasium as gym
import matplotlib.pyplot as plt

def train_qlearning():
    try:
        environment = gym.make('Taxi-v3')
    except:
        environment = gym.make('Taxi-v4')
    total_episodes = 500
    alphas = np.linspace(0, 1.0, 3)  # alpha
    gammas = np.linspace(0, 1.0, 3)  # gamma
    print('LET US LEARN NOW!')
    total_results = np.zeros((len(alphas), len(gammas)))
    for i in range(len(alphas)):
        for j in range(len(gammas)):
            # Caution, restart the object so that the Q table and results are reset
            qlearning = QLearning(environment=environment)
            qlearning.learning_rate = alphas[i]
            qlearning.discount_rate = gammas[j]
            qlearning.train(total_episodes=total_episodes)
            total_results[i][j]=np.mean(qlearning.results)
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
