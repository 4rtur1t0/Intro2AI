from qlearning_DQN.qlearning_DQN import QLearningDQN
import gymnasium as gym
from tqdm import tqdm


def set_experiments():
    experiments = [{'exp_name': 'Epsilon percentage 1',
                     'epsilon_percentage': 0.05},
                   {'exp_name': 'Epsilon percentage 2',
                    'epsilon_percentage': 0.1},
                   {'exp_name': 'Epsilon percentage 3',
                      'epsilon_percentage': 0.25},
                   {'exp_name': 'Epsilon percentage 4',
                     'epsilon_percentage': 0.5},
                   {'exp_name': 'Epsilon percentage 5',
                    'epsilon_percentage': 0.75},
                   {'exp_name': 'Epsilon percentage 6',
                    'epsilon_percentage': 0.95},
                   {'exp_name': 'Miopic/visionary 1',
                     'gamma': 0.1},
                   {'exp_name': 'Miopic/visionary 2',
                     'gamma': 0.3},
                   {'exp_name': 'Miopic/visionary 3',
                     'gamma': 0.5},
                   {'exp_name': 'Miopic/visionary 4 ',
                     'gamma': 0.99},
                   {'exp_name': 'MLP size 1_(8,8) g0.99',
                     'hidden_layer_sizes': (8, 8)},
                   {'exp_name': 'MLP size 2_(64,64) g0.99',
                     'hidden_layer_sizes': (64, 64)},
                   {'exp_name': 'MLP size 3_(128,128) g0.99',
                     'hidden_layer_sizes': (128, 128)},
                   {'exp_name': 'MLP size 1_(8,8) g0.5',
                     'gamma': 0.5,
                     'hidden_layer_sizes': (8, 8)},
                   {'exp_name': 'MLP size 2_(64,64) g0.5',
                     'gamma': 0.5,
                     'hidden_layer_sizes': (64, 64)},
                   {'exp_name': 'MLP size 3_(128,128) g0.5',
                     'gamma': 0.5,
                     'hidden_layer_sizes': (128, 128)},
                   {'exp_name': 'Replay buffer size 1',
                    'replay_buffer_size': 1},
                   {'exp_name': 'Replay buffer size 100',
                     'replay_buffer_size': 100},
                   {'exp_name': 'Replay buffer size 1000',
                     'replay_buffer_size': 1000},
                   {'exp_name': 'Replay buffer size 10000',
                     'replay_buffer_size': 10000},
                   {'exp_name': 'Replay buffer size 50000',
                     'replay_buffer_size': 50000},
                   {'exp_name': 'Target update freq 1',
                    'target_update_freq': 1},
                   {'exp_name': 'Target update freq 10',
                    'target_update_freq': 10},
                   {'exp_name': 'Target update freq 100',
                     'target_update_freq': 100},
                   {'exp_name': 'Target update freq 250',
                     'target_update_freq': 250},
                   {'exp_name': 'Target update freq 500',
                     'target_update_freq': 500},
                   {'exp_name': 'Target update freq 1000',
                    'target_update_freq': 1000},
                   {'exp_name': 'Target update freq 10000',
                    'target_update_freq': 10000}]
    return experiments

def train_deep_qlearning():
    try:
        environment = gym.make("LunarLander-v3")
    except Exception:
        environment = gym.make("LunarLander-v2")
    experiments = set_experiments()
    total_episodes_train = 1000
    total_episodes_test = 500
    repetitions = 3
    pbar = tqdm(total=repetitions*len(experiments), desc='Optimización DQN', colour='green')
    for experiment in experiments:
        for i in range(repetitions):
            pbar.update(1)
            qlearning = QLearningDQN(environment=environment, params=experiment)
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
    train_deep_qlearning()
