from qlearning.qlearning import QLearning
import gymnasium as gym
import matplotlib.pyplot as plt

def train_qlearning():
    # use render_mode="human" para observar el entorno gráficamente
    try:
        environment = gym.make('Taxi-v3')
    except:
        environment = gym.make('Taxi-v4')
    params = {'alpha': 0.5, 'gamma': 1.0}
    qlearning = QLearning(environment=environment, params=params)
    print('LET US LEARN NOW!')
    qlearning.train(total_episodes=500)
    qlearning.save_q_table(filename='qtable_taxi.npy')


if __name__ == "__main__":
    train_qlearning()
