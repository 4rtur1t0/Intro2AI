from qlearning.qlearning import QLearning
import gymnasium as gym

def train_qlearning():
    # use render_mode="human" para observar el entorno gráficamente
    try:
        environment = gym.make('Taxi-v3')
    except:
        environment = gym.make('Taxi-v4')
    qlearning = QLearning(environment=environment)
    print('LET US LEARN NOW!')
    qlearning.train(total_episodes=300)
    qlearning.save_q_table(filename='qtable_taxi.npy')

if __name__ == "__main__":
    train_qlearning()
