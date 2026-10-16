from qlearning.qlearning import QLearning
import gymnasium as gym

def test_qlearning():
    render_mode = "human"
    # render_mode = None
    try:
        environment = gym.make('Taxi-v3', render_mode=render_mode)
    except:
        environment = gym.make('Taxi-v4', render_mode=render_mode)
    qlearning = QLearning(environment=environment)
    # qlearning.create_random_q_table()
    qlearning.read_q_table(filename='qtable_taxi.npy')
    print('Testing mode!')
    qlearning.test(total_episodes=10)

if __name__ == "__main__":
    test_qlearning()
