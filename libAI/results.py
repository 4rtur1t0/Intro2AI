"""
    A solution for Deep QLearning.
    Q learning using a neural network to approximate Q
"""
import numpy as np
import json
import os
import datetime
import matplotlib.pyplot as plt
#from pyparsing import results


class Results():
    def __init__(self, type_result, params={}):
        self.params = params
        self.type_result = type_result
        self.results = []
        #self.recent_rewards = []

    def append_data(self, episode, total_reward, epsilon):
        self.results.append([episode, total_reward, epsilon])
        #self.recent_rewards.append(total_reward)

    #def avg_window_rewards(self, avg_window):
    #    return np.mean(self.recent_rewards[-self.avg_window:])

    def print_info(self, prelude='TRAIN', avg_window=100, flush_each=50):
        episode = self.results[-1][0]
        total_reward = self.results[-1][1]
        epsilon = self.results[-1][2]
        results = np.array(self.results)
        results = results[-avg_window:, 1]
        avg_reward = results.mean()
        print(f"{prelude} episode {episode:4d} | Total reward: {total_reward:6.1f} | Avg (100): {avg_reward:6.1f} | Epsilon: {epsilon:.2f}",
            end="\r", flush=True)
        if episode % flush_each == 0:
             print()

    def mean(self):
        res = np.array(self.results)
        return res.mean(axis=0)

    def reset_results(self):
        self.results = []

    def plot(self):
        results = np.array(self.results)
        plt.figure()
        # plt.plot(range(len(results)), , label="total reward")
        plt.plot(results[:, 0], results[:, 1], label="Total reward")
        plt.plot(results[:, 0], results[:, 2], label="100*epsilon")
        plt.xlabel("Episode")
        plt.ylabel("Reward/epsilon")
        plt.legend()
        plt.show()

    def save(self, experiment_name=None):
        """
        Save results as json
        :param experiment_name:
        :param type_result:
        :return:
        """
        now = datetime.datetime.now()
        time_string = now.strftime("%Y%m%d_%H%M%S")
        experiment_filename = f"results/{self.type_result}/{time_string}.json"
        # keep only the serializable params
        serializable_params = {k: v for k, v in self.params.items() if is_json_serializable(v)}
        experiment_data = {
            "experiment_name": experiment_name,
            "time_string": time_string,
            "experiment_filename": experiment_filename,
            "params": serializable_params,
            "episodes": [r[0] for r in self.results],
            "rewards": [r[1] for r in self.results],
            "100*epsilon": [r[2] for r in self.results],
            "global_mean_reward": np.mean([r[1] for r in self.results]),
            "global_mean_variance": np.std([r[1] for r in self.results])
        }
        # Ensure directory exists before writing
        os.makedirs(f"results/{self.type_result}", exist_ok=True)
        with open(experiment_filename, "w") as f:
            json.dump(experiment_data, f)

def is_json_serializable(value):
    try:
        json.dumps(value)
        return True
    except (TypeError, OverflowError):
        return False



