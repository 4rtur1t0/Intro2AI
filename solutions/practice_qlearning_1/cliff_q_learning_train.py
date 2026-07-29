from qlearning.qlearning import QLearning
import gymnasium as gym

def train_qlearning():
    # use render_mode="human" para observar el entorno gráficamente
    environment = gym.make('CliffWalking-v1')
    qlearning = QLearning(environment=environment)
    qlearning.learning_rate = 0.5  # alpha
    qlearning.discount_rate = 0.99  # gamma
    print('LET US LEARN NOW!')
    qlearning.train(total_episodes=100)
    # guardamos la tabla en disco
    qlearning.save_q_table(filename='qtable.npy')
    qlearning.results.plot_data()

if __name__ == "__main__":
    train_qlearning()
