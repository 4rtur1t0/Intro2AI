from qlearning.qlearning import QLearning
import gymnasium as gym

def train_qlearning():
    # use render_mode="human" para observar el entorno gráficamente
    environment = gym.make('CliffWalking-v1', max_episode_steps=200)
    params = {'alpha': 0.5, 'gamma': 0.5, 'percentage_target': 0.25}
    qlearning = QLearning(environment=environment, params=params)
    print('LET US LEARN NOW!')
    qlearning.train(total_episodes=100)
    # guardamos la tabla Q en disco
    qlearning.save_q_table(filename='qtable_cliff.npy')

if __name__ == "__main__":
    train_qlearning()
