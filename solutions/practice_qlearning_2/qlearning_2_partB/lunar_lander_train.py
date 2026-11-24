from qlearning_DQN.qlearning_DQN import QLearningDQN
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import datetime
import os
import json


def save(results, params):
    """
    Save results as json
    :param experiment_name:
    :param type_result:
    :return:
    """
    now = datetime.datetime.now()
    time_string = now.strftime("%Y%m%d_%H%M%S")
    experiment_filename = f"results/test/{time_string}.json"
    experiment_data = {
        "time_string": time_string,
        "experiment_filename": experiment_filename,
        "params": params,
        "rewards": results,
        "mean_reward": np.mean(results),
        "mean_variance": np.std(results)
    }
    # Ensure directory exists before writing
    os.makedirs(f"results/test", exist_ok=True)
    with open(experiment_filename, "w") as f:
        json.dump(experiment_data, f)

def plot_results(rewards_train, rewards_inline_test, epsilons, filter=False,window1=10, window2=5):
    if filter:
        rewards_train = np.convolve(rewards_train, np.ones(window1)/window1, mode='valid')
        rewards_inline_test = np.convolve(rewards_inline_test, np.ones(window2)/window2, mode='valid')
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


def train_deep_qlearning():
    # render_mode = 'human'
    render_mode = None
    try:
        environment = gym.make("LunarLander-v3", render_mode=render_mode)
    except Exception:
        environment = gym.make("LunarLander-v2", render_mode=render_mode)
    total_episodes_train = 300
    params = {'gamma': 0.99,
              'n_episodes_train': total_episodes_train,
              'epsilon_percentage': 0.25,
              'hidden_layer_sizes': (64, 64),
              'batch_size': 128,
              'target_update_freq': 250,
              'training_tests': (1500, 5)}
    qlearning = QLearningDQN(environment=environment, params=params)
    rewards_train, rewards_inline_test, epsilons = qlearning.train(total_episodes=total_episodes_train)
    qlearning.save_model(filename='qlearning_DQN_lunar_lander.pkl')
    plot_results(rewards_train, rewards_inline_test, epsilons, filter=True)
    # optional, TEST AND SAVE TO FILE
    total_episodes_test = 50
    results_test = qlearning.test(total_episodes=total_episodes_test)
    save(results_test, params)

if __name__ == "__main__":
    train_deep_qlearning()
