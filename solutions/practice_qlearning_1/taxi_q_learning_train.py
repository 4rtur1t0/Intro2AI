from qlearning.qlearning import QLearning
import gymnasium as gym

def train_qlearning():
    # use render_mode="human" para observar el entorno gráficamente
    environment = gym.make('Taxi-v4')
    qlearning = QLearning(environment=environment)
    qlearning.learning_rate = 0.5  # alpha
    qlearning.discount_rate = 0.9  # gamma
    print('LET US LEARN NOW!')
    qlearning.train(total_episodes=500)
    qlearning.save_q_table(filename='qtable_taxi.npy')

if __name__ == "__main__":
    train_qlearning()
