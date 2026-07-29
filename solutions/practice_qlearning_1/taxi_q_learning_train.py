from qlearning.qlearning import QLearning
import gymnasium as gym


def train_qlearning():
    # use render_mode="human" para observar el entorno gráficamente
    environment = gym.make('Taxi-v3')
    qlearning = QLearning(environment=environment)
    print('LET US LEARN NOW!')
    qlearning.train(total_episodes=15000)
    # guardamos la tabla en disco
    qlearning.save_q_table(filename='qtable_taxi.npy')
    qlearning.results.plot_data()

if __name__ == "__main__":
    train_qlearning()
