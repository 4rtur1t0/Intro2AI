from qlearning.qlearning import QLearning
import gymnasium as gym

def test_qlearning():
    environment = gym.make('CliffWalking-v1', render_mode="human")
    qlearning = QLearning(environment=environment)
    qlearning.load_random_q_table()
    # qlearning.load_q_table(filename='qtable.npy')
    qlearning.save_q_table('qtable_random.npy')
    print('Testing mode!')
    # qlearning.test(total_episodes=1)

if __name__ == "__main__":
    test_qlearning()
