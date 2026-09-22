from qlearning_discrete.qlearning_discrete import QLearningD
import gymnasium as gym


def set_experiments():
    # total_episodes = 10000
    experiments = [ {'exp_name': 'Explor./exploit. 1',
                     'alpha': 0.5,
                     'gamma': 0.99,
                     'epsilon_percentage': 0.1},
                    {'exp_name': 'Explor./exploit. 2',
                     'alpha': 0.5,
                     'gamma': 0.99,
                    'epsilon_percentage': 0.25},
                     {'exp_name': 'Explor./exploit. 3',
                      'alpha': 0.5,
                      'gamma': 0.99,
                      'epsilon_percentage': 0.5},
                     {'exp_name': 'Explor./exploit. 4',
                      'alpha': 0.5,
                      'gamma': 0.99,
                      'epsilon_percentage': 0.75},
                     {'exp_name': 'Explor./exploit. 5',
                      'alpha': 0.5,
                      'gamma': 0.99,
                      'epsilon_percentage': 0.95},
                     {'exp_name': 'Explor./exploit. 6',
                      'alpha': 0.5,
                      'gamma': 0.99,
                      'epsilon_percentage': 1.0},
                   {'exp_name': 'Explor./exploit. 7',
                    'alpha': 0.5,
                    'gamma': 0.99,
                    'epsilon_max': 0.1,
                    'epsilon_min': 0.1,
                    'epsilon_percentage': 1.0},
                   {'exp_name': 'Explor./exploit. 8',
                    'alpha': 0.5,
                    'gamma': 0.99,
                    'epsilon_max': 0.2,
                    'epsilon_min': 0.2,
                    'epsilon_percentage': 1.0},
                    {'exp_name': 'Explor./exploit. 9',
                     'alpha': 0.5,
                     'gamma': 0.99,
                     'epsilon_max': 0.5,
                     'epsilon_min': 0.5,
                     'epsilon_percentage': 1.0},
                    {'exp_name': 'Learning factor 1',
                     'alpha': 0.1,
                     'gamma': 0.99,
                     'epsilon_percentage': 0.25},
                    {'exp_name': 'Learning factor 2',
                     'alpha': 0.2,
                     'gamma': 0.99,
                     'epsilon_percentage': 0.25},
                    {'exp_name': 'Learning factor 3',
                     'alpha': 0.5,
                     'gamma': 0.99,
                     'epsilon_percentage': 0.25},
                    {'exp_name': 'Learning factor 4',
                     'alpha': 0.7,
                     'gamma': 0.99,
                     'epsilon_percentage': 0.25},
                    {'exp_name': 'Learning factor 5',
                     'alpha': 1.0,
                     'gamma': 0.99,
                     'epsilon_percentage': 0.25}
                    ]
    return experiments

def train_discrete_qlearning_batch():
    try:
        environment = gym.make("LunarLander-v3")
    except Exception:
        environment = gym.make("LunarLander-v2")
    experiments = set_experiments()
    total_episodes_train = 10000
    total_episodes_test = 1000
    repetitions = 5
    for experiment in experiments:
        for i in range(repetitions):
            qlearning = QLearningD(environment=environment, params=experiment)
            print(30*'#')
            print('TRAIN ' + experiment['exp_name'])
            print(30*'#')
            # TRAIN!
            results = qlearning.train(total_episodes=total_episodes_train)
            results.save(experiment_name=experiment['exp_name'])
            print(30 * '#')
            print('TEST ' + experiment['exp_name'])
            print(30 * '#')
            results = qlearning.test(total_episodes=total_episodes_test)
            results.save(experiment_name=experiment['exp_name'])

if __name__ == "__main__":
    train_discrete_qlearning_batch()
