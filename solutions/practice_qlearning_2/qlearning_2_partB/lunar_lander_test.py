from qlearning_DQN.qlearning_DQN import QLearningDQN
import gymnasium as gym

def test_deep_qlearning():
    try:
        environment = gym.make("LunarLander-v3", render_mode='human')
    except Exception:
        environment = gym.make("LunarLander-v2", render_mode='human')
    qlearning = QLearningDQN(environment=environment)
    qlearning.read_model(filename='mlpregressor_qtable_lunar_lander.pkl')
    print('LET US SEE WHAT WAS LEARNT NOW!')
    qlearning.test(total_episodes=20)

if __name__ == "__main__":
    test_deep_qlearning()
