from qlearning_discrete.qlearning_discrete import QLearningD
import gymnasium as gym

def train_qlearning():
    # use render_mode="human" para observar el entorno gráficamente
    try:
        environment = gym.make("LunarLander-v3")
    except Exception:
        environment = gym.make("LunarLander-v2")
    qlearning = QLearningD(environment=environment)
    qlearning.read_q_table(filename='qtable_lunar_lander.pkl')
    print('LET US LEARN NOW!')
    qlearning.train(total_episodes=15000)
    qlearning.save_q_table(filename='qtable_lunar_lander.pkl')

if __name__ == "__main__":
    train_qlearning()
