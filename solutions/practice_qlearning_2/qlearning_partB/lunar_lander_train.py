from qlearning_DQN.qlearning_DQN import QLearningDQN
import gymnasium as gym

def train_qlearning():
    # use render_mode="human" para observar el entorno gráficamente
    try:
        environment = gym.make("LunarLander-v3")
    except Exception:
        environment = gym.make("LunarLander-v2")
    qlearning = QLearningDQN(environment=environment)
    qlearning.read_q_table(filename='qtable_lunar_lander.pkl')
    print('LET US LEARN NOW!')
    qlearning.train(total_episodes=15000)
    qlearning.save_q_table(filename='qtable_lunar_lander.pkl')

if __name__ == "__main__":
    train_qlearning()
