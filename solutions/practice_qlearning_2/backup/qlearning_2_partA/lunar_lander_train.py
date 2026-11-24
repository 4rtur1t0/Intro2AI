from qlearning_discrete.qlearning_discrete import QLearningD
import gymnasium as gym

def train_qlearning():
    try:
        environment = gym.make("LunarLander-v3")
    except Exception:
        environment = gym.make("LunarLander-v2")
    total_episodes_train = 3000
    params = {'gamma': 1.0, 'alpha': 0.5,
              'epsilon_percentage': 0.25, 'training_tests': (100, 10)}
    qlearning = QLearningD(environment=environment, params=params)
    qlearning.read_model(filename='qtable_lunar_lander.pkl')
    print('LET US LEARN NOW!')
    results = qlearning.train(total_episodes=total_episodes_train)
    qlearning.save_model(filename='qtable_lunar_lander.pkl')
    # qlearning.save_results(experiment_name=str(params), type_result='train')
    results.save(experiment_name=str(params))
    results.plot()

if __name__ == "__main__":
    train_qlearning()
