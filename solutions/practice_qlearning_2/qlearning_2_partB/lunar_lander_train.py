from qlearning_DQN.qlearning_DQN import QLearningDQN
import gymnasium as gym

def train_deep_qlearning():
    try:
        environment = gym.make("LunarLander-v3")
    except Exception:
        environment = gym.make("LunarLander-v2")
    qlearning = QLearningDQN(environment=environment)
    # podemos leer el último modelo guardado para seguir entrenando sobre él
    qlearning.read_model(filename='mlpregressor_qtable_lunar_lander.pkl')
    print('LET US LEARN NOW!')
    qlearning.train(total_episodes=800)
    qlearning.save_model(filename='mlpregressor_qtable_lunar_lander.pkl')

if __name__ == "__main__":
    train_deep_qlearning()
