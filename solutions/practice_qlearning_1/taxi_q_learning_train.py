import matplotlib.pyplot as plt
from qlearning.qlearning import QLearning
import gymnasium as gym

def train_qlearning():
    # use render_mode="human" para observar el entorno gráficamente
    try:
        environment = gym.make('Taxi-v3')
    except:
        environment = gym.make('Taxi-v4')
    params = {'alpha': 0.1, 'gamma': 1.0}
    qlearning = QLearning(environment=environment, params=params)
    print('LET US LEARN NOW!')
    results = qlearning.train(total_episodes=2000)
    qlearning.save_q_table(filename='qtable_taxi.npy')
    plt.figure()
    plt.plot(results)
    plt.ylabel('Mean reward')
    plt.xlabel('Episodes (inline tests)')
    plt.show()

if __name__ == "__main__":
    train_qlearning()
