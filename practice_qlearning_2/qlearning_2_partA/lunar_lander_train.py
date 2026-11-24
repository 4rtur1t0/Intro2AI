from qlearning_discrete.qlearning_discrete import QLearningD
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

def plot_results(rewards_train, rewards_inline_test, epsilons, filter=False,window1=500, window2=30):
    if filter:
        rewards_train = np.convolve(np.array(rewards_train), np.ones(window1)/window1, 'valid')
        rewards_inline_test = np.convolve(rewards_inline_test, np.ones(window2) / window2, mode='valid')
    fig, axs = plt.subplots(3)
    fig.suptitle('Episode rewards')
    axs[0].plot(rewards_train)
    axs[0].set_ylabel('Rewards train')
    axs[1].plot(epsilons)
    axs[1].set_ylabel('Epsilon')
    # TODO! Has de fer que siguen comparables les gràfiques.
    x = np.linspace(0, len(rewards_train), len(rewards_inline_test))
    axs[2].plot(x, rewards_inline_test)
    axs[2].set_ylabel('Rewards inline test')
    axs[2].set_xlabel('Episodes')
    plt.show()


def train_qlearning():
    try:
        environment = gym.make("LunarLander-v3")
    except Exception:
        environment = gym.make("LunarLander-v2")
    total_episodes_train = 30000
    params = {'gamma': 0.99, 'alpha': 0.5,
              'epsilon_percentage': 0.25, 'training_tests': (100, 10)}
    qlearning = QLearningD(environment=environment, params=params)
    print('LET US LEARN NOW!')
    rewards_train, rewards_inline_test, epsilons = qlearning.train(total_episodes=total_episodes_train)
    qlearning.save_model(filename='qtable_lunar_lander.pkl')
    plot_results(rewards_train, rewards_inline_test, epsilons, filter=True)

if __name__ == "__main__":
    train_qlearning()
