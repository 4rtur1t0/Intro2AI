from qlearning.qlearning import QLearning
import gymnasium as gym

def train_qlearning():
    # use render_mode="human" para observar el entorno gráficamente
    environment = gym.make('Taxi-v3')
    qlearning = QLearning(environment=environment)
    qlearning.learning_rate = 1.0  # alpha
    qlearning.discount_rate = 1.0  # gamma
    print('LET US LEARN NOW!')
    qlearning.train(total_episodes=5000)
    qlearning.save_q_table(filename='qtable_taxi.npy')

if __name__ == "__main__":
    train_qlearning()
