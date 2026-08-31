from qlearning_discrete.qlearning_discrete import QLearningD
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

def plot_results(qlearning):
    results = np.array(qlearning.results)
    plt.figure()
    plt.plot(results[:,0], results[:,1], label="Total reward (inline test average)")
    plt.plot(results[:,0], results[:,2], label="100*epsilon")
    plt.xlabel("Episode")
    plt.ylabel("Reward/epsilon")
    plt.legend()
    plt.show()


def train_qlearning():
    try:
        environment = gym.make("LunarLander-v3")
    except Exception:
        environment = gym.make("LunarLander-v2")
    params = {'gamma': 1.0, 'alpha': 0.5,
              'epsilon_percentage': 0.5, 'training_tests': (1000, 100)}
    qlearning = QLearningD(environment=environment, params=params)
    #qlearning.read_q_table(filename='qtable_lunar_lander.pkl')
    print('LET US LEARN NOW!')
    qlearning.train(total_episodes=300000)
    qlearning.save_q_table(filename='qtable_lunar_lander.pkl')
    qlearning.save_results(experiment_name=str(params), type_result='train')
    plot_results(qlearning)

if __name__ == "__main__":
    train_qlearning()
