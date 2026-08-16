from qlearning_discrete.qlearning_discrete import QLearningD
import gymnasium as gym

def test_qlearning():
    try:
        environment = gym.make("LunarLander-v3", render_mode="human")
    except Exception:
        environment = gym.make("LunarLander-v2", render_mode="human")
    qlearningd = QLearningD(environment=environment)
    qlearningd.read_q_table(filename='qtable_lunar_lander.pkl')
    print('Testing mode!')
    qlearningd.test(total_episodes=10)

if __name__ == "__main__":
    test_qlearning()
