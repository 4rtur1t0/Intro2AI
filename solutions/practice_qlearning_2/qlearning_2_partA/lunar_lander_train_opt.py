"""
Se desea hacer una imagen con el resultado sobre test al entrenar con diferentes alphas y gammas.
"""
import numpy as np
from qlearning_discrete.qlearning_discrete import QLearningD
import gymnasium as gym
import matplotlib.pyplot as plt
from tqdm import tqdm

def train_qlearning():
    try:
        environment = gym.make("LunarLander-v3")
    except Exception:
        environment = gym.make("LunarLander-v2")
    total_episodes_train = 15000
    total_episodes_test = 2000
    repetitions = 5
    alphas = np.linspace(0.05, 1.0, 5)  # alpha
    gammas = np.linspace(0.05, 1.0, 5)  # gamma
    print('LET US LEARN NOW!')
    total_results = np.zeros((len(alphas), len(gammas)))
    pbar = tqdm(total=repetitions*len(alphas)*len(gammas), desc='Optimización discretized Q learning', colour='green')
    for i in range(len(alphas)):
        for j in range(len(gammas)):
            results = []
            for k in range(repetitions):
                pbar.update(1)
                # Caution, restart the object so that the Q table and results are reset
                # en este caso, no deseamos hacer tests inline
                params = {'alpha': alphas[i], 'gamma': gammas[j],
                          'training_tests': (1000, 0),
                          'percentage_target': 0.25}
                qlearning = QLearningD(environment=environment, params=params)
                qlearning.train(total_episodes=total_episodes_train)
                res = qlearning.test(total_episodes=total_episodes_test)
                results.append(res)
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
    ax.set_title("Reward medio en episodios de test")
    plt.xlabel("gamma")
    plt.ylabel("alfa")
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_qlearning()
