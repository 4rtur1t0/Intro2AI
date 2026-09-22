"""
    Se desea
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
    alphas = [0.1, 0.5]
    gammas = [0.1, 0.9]
    print('LET US LEARN NOW!')
    legends = []
    total_results = []
    for alpha in alphas:
        for gamma in gammas:
            print('Training on alpha: {}, gamma: {}'.format(alpha, gamma))
            # Caution, restart the object so that the Q table and results are reset
            params = {'alpha': alpha, 'gamma': gamma}
            qlearning = QLearning(environment=environment, params=params)
            # devuelve el resultado de los inline tests
            results = qlearning.train(total_episodes=1500)
            #results = qlearning.test(total_episodes=10)
            # se guardan en una lista
            total_results.append(results)
            legends.append('(alpha, gamma)=({}_{})'.format(alpha, gamma))
    total_results = np.array(total_results)
    # ploteamos los resultados
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
