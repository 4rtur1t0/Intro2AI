from qlearning_discrete.qlearning_discrete import QLearningD
import gymnasium as gym


def set_experiments():
    total_episodes = 10000
    experiments = [ {'exp_name': 'Explor./exploit. 1',
                    'total_episodes': total_episodes,
                     'alpha': 0.1,
                    'gamma': 0.99,
                    'epsilon_max': 1.0,
                    'epsilon_min': 0.01,
                    'epsilon_percentage': 0.1},
                    {'exp_name': 'Explor./exploit. 2',
                    'total_episodes': total_episodes,
                     'alpha': 0.1,
                    'gamma': 0.99,
                    'epsilon_max': 1.0,
                    'epsilon_min': 0.01,
                    'epsilon_percentage': 0.20},
                     {'exp_name': 'Explor./exploit. 3',
                    'total_episodes': total_episodes,
                      'alpha': 0.1,
                    'gamma': 0.99,
                    'epsilon_max': 1.0,
                    'epsilon_min': 0.01,
                    'epsilon_percentage': 0.5},
                   {'exp_name': 'Explor./exploit. 4',
                    'total_episodes': total_episodes,
                    'alpha': 0.1,
                    'gamma': 0.99,
                    'epsilon_max': 1.0,
                    'epsilon_min': 0.01,
                    'epsilon_percentage': 0.95},
                    {'exp_name': 'Explor./exploit. 5',
                     'total_episodes': total_episodes,
                     'alpha': 0.1,
                     'gamma': 0.99,
                     'epsilon_max': 0.01,
                     'epsilon_min': 0.01,
                     'epsilon_percentage': 1.0},
                    {'exp_name': 'Explor./exploit. 6',
                    'total_episodes': total_episodes,
                     'alpha': 0.1,
                    'gamma': 0.99,
                    'epsilon_max': 0.2,
                    'epsilon_min': 0.2,
                    'epsilon_percentage': 1.0},
                   {'exp_name': 'Explor./exploit. 7',
                    'total_episodes': total_episodes,
                    'alpha': 0.1,
                    'gamma': 0.99,
                    'epsilon_max': 0.5,
                    'epsilon_min': 0.5,
                    'epsilon_percentage': 1.0},
                   {'exp_name': 'Explor./exploit. 8',
                    'total_episodes': total_episodes,
                    'alpha': 0.1,
                    'gamma': 0.99,
                    'epsilon_max': 0.95,
                    'epsilon_min': 0.95,
                    'epsilon_percentage': 1.0}]
    return experiments

def train_discrete_qlearning_batch():
    try:
        environment = gym.make("LunarLander-v3")
    except Exception:
        environment = gym.make("LunarLander-v2")
    experiments = set_experiments()
    for experiment in experiments:
        qlearning = QLearningD(environment=environment, params=experiment)
        print(30*'#')
        print('TRAIN ' + experiment['exp_name'])
        print(30*'#')
        total_episodes = experiment['total_episodes']
        # TRAIN!
        qlearning.train(total_episodes=total_episodes)
        qlearning.save_results(experiment_name=experiment['exp_name'],
                               type_result='train')
        print(30 * '#')
        print('TEST ' + experiment['exp_name'])
        print(30 * '#')
        qlearning.test(total_episodes=1000)
        qlearning.save_results(experiment_name=experiment['exp_name'],
                               type_result='test')


if __name__ == "__main__":
    train_discrete_qlearning_batch()
