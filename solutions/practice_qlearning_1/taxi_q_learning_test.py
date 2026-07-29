from qlearning.qlearning import QLearning
import gymnasium as gym

def test_qlearning():
    environment = gym.make('Taxi-v4', render_mode="human")
    qlearning = QLearning(environment=environment)
    qlearning.read_q_table(filename='qtable_taxi.npy')
    print('Testing mode!')
    qlearning.test(total_episodes=10)

if __name__ == "__main__":
    test_qlearning()
