from qlearning_discrete.qlearning_discrete import QLearningD
import gymnasium as gym

def test_qlearning():
    render_mode = None #'human'
    episodes_test = 200
    try:
        environment = gym.make("LunarLander-v3", render_mode=render_mode)
    except Exception:
        environment = gym.make("LunarLander-v2", render_mode=render_mode)
    qlearningd = QLearningD(environment=environment)
    qlearningd.read_model(filename='qtable_lunar_lander.pkl')
    print('Testing mode!')
    results = qlearningd.test(total_episodes=episodes_test)
    results.save(experiment_name='test')
    results.plot()

if __name__ == "__main__":
    test_qlearning()
