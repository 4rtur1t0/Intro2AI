from qlearning_discrete.qlearning_discrete import QLearningD
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

def plot_results(qlearning):
    results = np.array(qlearning.results)
    plt.figure()
    plt.plot(range(len(results)), results[:,0], label="Total reward")
    plt.plot(range(len(results)), results[:,1], label="Total reward (average)")
    plt.plot(range(len(results)), results[:,2], label="100*epsilon")
    plt.plot(range(len(results)), results[:,3], label="N visited states (x1000)")
    plt.legend()
    plt.show()


def train_qlearning():
    # use render_mode="human" para observar el entorno gráficamente
    try:
        environment = gym.make("LunarLander-v3")
    except Exception:
        environment = gym.make("LunarLander-v2")
    qlearning = QLearningD(environment=environment)
    #qlearning.read_q_table(filename='qtable_lunar_lander.pkl')
    print('LET US LEARN NOW!')
    qlearning.gamma = 1.0
    qlearning.alpha = 0.1
    qlearning.train(total_episodes=150000)
    qlearning.save_q_table(filename='qtable_lunar_lander.pkl')
    qlearning.save_results()
    plot_results(qlearning)

if __name__ == "__main__":
    train_qlearning()
