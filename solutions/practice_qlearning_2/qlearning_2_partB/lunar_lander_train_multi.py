from qlearning_DQN.qlearning_DQN import QLearningDQN
import gymnasium as gym


def set_experiments():
    total_episodes = 1000
    experiments = [

                   # {'exp_name': 'Miopic/visionary 1',
                   #  'total_episodes': total_episodes,
                   #  'gamma': 0.1,
                   #  'epsilon_max': 1.0,
                   #  'epsilon_min': 0.01,
                   #  'epsilon_percentage': 0.2,
                   #  'hidden_layer_sizes': (64, 64)},
                   # {'exp_name': 'Miopic/visionary 2',
                   #  'total_episodes': total_episodes,
                   #  'gamma': 0.3,
                   #  'epsilon_max': 1.0,
                   #  'epsilon_min': 0.01,
                   #  'epsilon_percentage': 0.2,
                   #  'hidden_layer_sizes': (64, 64)},
                   # {'exp_name': 'Miopic/visionary 3',
                   #  'total_episodes': total_episodes,
                   #  'gamma': 0.5,
                   #  'epsilon_max': 1.0,
                   #  'epsilon_min': 0.01,
                   #  'epsilon_percentage': 0.2,
                   #  'hidden_layer_sizes': (64, 64)},
                   # {'exp_name': 'Miopic/visionary 4',
                   #  'total_episodes': total_episodes,
                   #  'gamma': 0.99,
                   #  'epsilon_max': 1.0,
                   #  'epsilon_min': 0.01,
                   #  'epsilon_percentage': 0.5,
                   #  'hidden_layer_sizes': (64, 64)}
                   {'exp_name': 'MLP size 1_(8,8)',
                    'total_episodes': total_episodes,
                    'gamma': 0.99,
                    'epsilon_max': 1.0,
                    'epsilon_min': 0.01,
                    'epsilon_percentage': 0.2,
                    'hidden_layer_sizes': (8, 8)},
                   {'exp_name': 'MLP size 2_(64,64)',
                    'total_episodes': total_episodes,
                    'gamma': 0.99,
                    'epsilon_max': 1.0,
                    'epsilon_min': 0.01,
                    'epsilon_percentage': 0.2,
                    'hidden_layer_sizes': (64, 64)},
                   {'exp_name': 'MLP size 3_(256,256,256)',
                    'total_episodes': total_episodes,
                    'gamma': 0.99,
                    'epsilon_max': 1.0,
                    'epsilon_min': 0.01,
                    'epsilon_percentage': 0.2,
                    'hidden_layer_sizes': (256, 256, 256)}
                    ]
    return experiments

def train_deep_qlearning():
    try:
        environment = gym.make("LunarLander-v3")
    except Exception:
        environment = gym.make("LunarLander-v2")
    experiments = set_experiments()
    for experiment in experiments:
        qlearning = QLearningDQN(environment=environment, params=experiment)
        print(30*'#')
        print('TRAIN ' + experiment['exp_name'])
        print(30*'#')
        total_episodes = experiment['total_episodes']
        # TRAIN!
        qlearning.train(total_episodes=total_episodes)
        #qlearning.save_model(filename='mlpregressor_qtable_lunar_lander.pkl')
        qlearning.save_results(experiment_name=experiment['exp_name'],
                               type_result='train')
        qlearning.reset_results()
        print(30 * '#')
        print('TEST ' + experiment['exp_name'])
        print(30 * '#')
        qlearning.test(total_episodes=1000, save_results=True)
        qlearning.save_results(experiment_name=experiment['exp_name'],
                               type_result='test')
        environment.close()


if __name__ == "__main__":
    train_deep_qlearning()
