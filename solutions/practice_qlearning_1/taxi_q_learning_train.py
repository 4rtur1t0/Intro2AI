import matplotlib.pyplot as plt
from qlearning.qlearning import QLearning
import gymnasium as gym

def test():
    render_mode = "human"
    # render_mode = None
    try:
        environment = gym.make('Taxi-v3', render_mode=render_mode)
    except:
        environment = gym.make('Taxi-v4', render_mode=render_mode)
    qlearning = QLearning(environment=environment)
    qlearning.read_q_table(filename='qtable_taxi.npy')
    print('Testing mode!')
    qlearning.test(total_episodes=10)


def train_qlearning():
    # use render_mode="human" para observar el entorno gráficamente
    render_mode = None
    # render_mode = "human"
    try:
        environment = gym.make('Taxi-v3', render_mode=render_mode)
    except:
        environment = gym.make('Taxi-v4', render_mode=render_mode)
    params = {'alpha': 0.9, 'gamma': 0.1}
    qlearning = QLearning(environment=environment, params=params)
    print('LET US LEARN NOW!')
    results = qlearning.train(total_episodes=10000)
    qlearning.save_q_table(filename='qtable_taxi.npy')

    plt.figure()
    plt.plot(results)
    plt.ylabel('Mean reward')
    plt.xlabel('Episodes (inline tests)')
    plt.show()

    test()

if __name__ == "__main__":
    train_qlearning()
