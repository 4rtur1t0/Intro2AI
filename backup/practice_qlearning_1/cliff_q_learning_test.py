from qlearning.qlearning import QLearning
import gymnasium as gym

def test_qlearning():
    environment = gym.make('CliffWalking-v1', render_mode='human', max_episode_steps=200)
    qlearning = QLearning(environment=environment)
    qlearning.create_random_q_table()
    #qlearning.read_q_table(filename='qtable.npy')
    print('Testing mode!')
    qlearning.test(total_episodes=10)

if __name__ == "__main__":
    test_qlearning()
