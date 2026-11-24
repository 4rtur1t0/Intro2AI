from qlearning_DQN.qlearning_DQN import QLearningDQN
import gymnasium as gym

def test_deep_qlearning():
    render_mode = 'human'
    # render_mode = None
    try:
        environment = gym.make("LunarLander-v3", render_mode=render_mode)
    except Exception:
        environment = gym.make("LunarLander-v2", render_mode=render_mode)
    qlearning = QLearningDQN(environment=environment)
    qlearning.read_model(filename='qlearning_DQN_lunar_lander.pkl')
    print('LET US SEE WHAT WAS LEARNT NOW!')
    results = qlearning.test(total_episodes=20)
    results.save(experiment_name='test')
    results.plot()

if __name__ == "__main__":
    test_deep_qlearning()
