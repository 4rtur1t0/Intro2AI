from qlearning.qlearning import QLearning
import gymnasium as gym

def test_qlearning():
    environment = gym.make('CliffWalking-v1', render_mode="human")
    qlearning = QLearning(environment=environment)
    qlearning.load_random_q_table()
    print('Testing mode!')
    qlearning.test(total_episodes_test=1)

if __name__ == "__main__":
    test_qlearning()
