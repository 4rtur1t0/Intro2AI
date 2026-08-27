from qlearning.qlearning import QLearning
import gymnasium as gym

def test_qlearning():
    try:
        environment = gym.make('Taxi-v3', render_mode="human")
    except:
        environment = gym.make('Taxi-v4', render_mode="human")
    qlearning = QLearning(environment=environment)
    #qlearning_discrete.create_random_q_table()
    qlearning.read_q_table(filename='qtable_taxi.npy')
    print('Testing mode!')
    qlearning.test(total_episodes=10)

if __name__ == "__main__":
    test_qlearning()
