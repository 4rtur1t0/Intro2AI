from qlearning.qlearning import QLearning
import gymnasium as gym

def test():
    environment = gym.make('CliffWalking-v1', max_episode_steps=200, render_mode='human')
    qlearning = QLearning(environment=environment)
    qlearning.read_q_table('qtable_cliff.npy')
    qlearning.test(total_episodes=1)


def train_qlearning():
    render_mode = None
    # render_mode = "human" # use render_mode="human" para observar el entorno gráficamente
    environment = gym.make('CliffWalking-v1', max_episode_steps=200, render_mode=render_mode)
    params = {'alpha': 1.0, 'gamma': 1.0, 'percentage_target': 0.25}
    qlearning = QLearning(environment=environment, params=params)
    print('LET US LEARN NOW!')
    qlearning.train(total_episodes=100)
    # guardamos la tabla Q en disco
    qlearning.save_q_table(filename='qtable_cliff.npy')

    # OPCIONAL! hacemos un mini-test visual para ver si aprende nuestro agente
    test()

if __name__ == "__main__":
    train_qlearning()
