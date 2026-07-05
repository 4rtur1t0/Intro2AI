from qlearning.qlearning import QLearning
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


#
# def plot_q_table(q_table):
#     n_rows = 4
#     n_cols = 12
#     q_table_2d = np.zeros((n_rows, n_cols))
#     for row in range(n_rows):
#         for col in range(n_cols):
#             index = row * n_cols + col
#             print(index)
#             action_values = q_table[index]
#             q_table_2d[row, col] = np.argmax(action_values)
#     print(q_table_2d)
#     plt.imshow(q_table_2d, interpolation='none')
#     plt.colorbar()
#     plt.title("Q-table for the cliff agent")
#     plt.show()


def train_qlearning():
    # use render_mode="human" para observar el entorno gráficamente
    environment = gym.make('CliffWalking-v1')
    qlearning = QLearning(environment=environment)
    print('LET US LEARN NOW!')
    qlearning.train(total_episodes_train=3000)
    # guardamos la tabla en disco
    qlearning.save_q_table(filename='qtable.npy')
    qlearning.results.plot_data()
    #plot_q_table(qlearning.q_table)
    #print('LET US USE OUR FULL KNOWLEDGE NOW!')
    #qlearning.test(total_episodes_test=10)
    #qlearning.results_test.plot_data()



if __name__ == "__main__":
    train_qlearning()
