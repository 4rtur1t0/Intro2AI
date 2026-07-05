from qlearning.qlearning import QLearning
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


def plot_q_table(q_table):
    n_rows = 4
    n_cols = 12
    q_table_2d = np.zeros((n_rows, n_cols))
    for row in range(n_rows):
        for col in range(n_cols):
            index = row * n_cols + col
            print(index)
            action_values = q_table[index]
            q_table_2d[row, col] = np.argmax(action_values)
    print(q_table_2d)
    plt.imshow(q_table_2d, interpolation='none')
    plt.colorbar()
    plt.title("Q-table for the cliff agent")
    plt.show()


def solve_qlearning():
    train_env = gym.make('FrozenLake-v1',desc=None, map_name="8x8",
                         is_slippery=False, success_rate=1.0/3.0,reward_schedule=(1, 0, 0))
    test_env = gym.make('FrozenLake-v1',desc=None, map_name="8x8",
                         is_slippery=False, success_rate=1.0/3.0,reward_schedule=(1, 0, 0), render_mode="human")
    qlearning = QLearning(train_env=train_env, test_env=test_env)
    print('LET US LEARN NOW!')
    qlearning.train(total_episodes_train=3000)
    qlearning.results_train.plot_data()
    #plot_q_table(pract_3_qlearning.q_table)
    print('LET US USE OUR FULL KNOWLEDGE NOW!')
    qlearning.test(total_episodes_test=10)
    qlearning.results_test.plot_data()



if __name__ == "__main__":
    solve_qlearning()
