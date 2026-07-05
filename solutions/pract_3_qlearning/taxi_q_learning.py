from qlearning.qlearning import QLearning
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


def plot_q_table(q_table, passenger_location, destination):
    human_readable_max_action_table = np.zeros((5, 5))
    for taxi_row in range(5):
        for taxi_col in range(5):
            index = (((taxi_row * 5 + taxi_col) * 5 + passenger_location) * 4
                     + destination)
            actions = q_table[index]
            human_readable_max_action_table[taxi_row, taxi_col] = np.argmax(actions)
    plt.imshow(human_readable_max_action_table, interpolation='none')
    plt.colorbar()
    plt.title("Q-table for a passenger location and destination")
    plt.show()


def solve_taxi():
    # Adjust as needed:
    # Usa render_mode="human" para depurar o visualizar el test
    train_env = gym.make("Taxi-v3")
    test_env = gym.make("Taxi-v3", render_mode="human")
    qlearning = QLearning(train_env=train_env, test_env=test_env)

    print('LET US LEARN NOW!')
    qlearning.train(total_episodes_train=30000)
    qlearning.results_train.plot_data()
    plot_q_table(q_table=qlearning.q_table, passenger_location=0, destination=1)
    print('LET US USE OUR FULL KNOWLEDGE NOW!')
    qlearning.test(total_episodes_test=30)
    qlearning.results_test.plot_data()



if __name__ == "__main__":
    solve_taxi()
