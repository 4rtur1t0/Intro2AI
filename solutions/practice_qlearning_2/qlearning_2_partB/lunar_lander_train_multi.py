from qlearning_DQN.qlearning_DQN import QLearningDQN
import gymnasium as gym
from tqdm import tqdm


def set_experiments():
    experiments = [ {'exp_name': 'Miopic/visionary 1',
                     'gamma': 0.1,
                     'hidden_layer_sizes': (64, 64)},
                    {'exp_name': 'Miopic/visionary 2',
                     'gamma': 0.3,
                     'hidden_layer_sizes': (64, 64)},
                    {'exp_name': 'Miopic/visionary 3',
                     'gamma': 0.5,
                     'hidden_layer_sizes': (64, 64)},
                    {'exp_name': 'Miopic/visionary 4 ',
                     'gamma': 0.99,
                     'hidden_layer_sizes': (64, 64)},
                    {'exp_name': 'MLP size 1_(8,8) g0.99',
                     'gamma': 0.99,
                     'hidden_layer_sizes': (8, 8)},
                    {'exp_name': 'MLP size 2_(64,64) g0.99',
                     'gamma': 0.99,
                     'hidden_layer_sizes': (64, 64)},
                    {'exp_name': 'MLP size 3_(128,128) g0.99',
                     'gamma': 0.99,
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
                    'gamma': 0.99,
                    'replay_buffer_size': 1,
                    'hidden_layer_sizes': (64, 64)},
                   {'exp_name': 'Replay buffer size 100',
                     'gamma': 0.99,
                     'replay_buffer_size': 100,
                     'hidden_layer_sizes': (64, 64)},
                    {'exp_name': 'Replay buffer size 1000',
                     'gamma': 0.99,
                     'replay_buffer_size': 1000,
                     'hidden_layer_sizes': (64, 64)},
                    {'exp_name': 'Replay buffer size 10000',
                     'gamma': 0.99,
                     'replay_buffer_size': 10000,
                     'hidden_layer_sizes': (64, 64)},
                    {'exp_name': 'Replay buffer size 50000',
                     'gamma': 0.99,
                     'replay_buffer_size': 50000,
                     'hidden_layer_sizes': (64, 64)},
                    {'exp_name': 'Target update freq 1',
                    'gamma': 0.99,
                    'target_update_freq': 1,
                    'hidden_layer_sizes': (64, 64)},
                   {'exp_name': 'Target update freq 10',
                    'gamma': 0.99,
                    'target_update_freq': 10,
                    'hidden_layer_sizes': (64, 64)},
                   {'exp_name': 'Target update freq 100',
                    'gamma': 0.99,
                    'target_update_freq': 100,
                    'hidden_layer_sizes': (64, 64)},
                   {'exp_name': 'Target update freq 250',
                    'gamma': 0.99,
                    'target_update_freq': 250,
                    'hidden_layer_sizes': (64, 64)},
                   {'exp_name': 'Target update freq 500',
                    'gamma': 0.99,
                    'target_update_freq': 500,
                    'hidden_layer_sizes': (64, 64)},
                   {'exp_name': 'Target update freq 1000',
                    'gamma': 1.0,
                    'target_update_freq': 1000,
                    'hidden_layer_sizes': (64, 64)},
                   {'exp_name': 'Target update freq 10000',
                    'gamma': 1.0,
                    'target_update_freq': 10000,
                    'hidden_layer_sizes': (64, 64)} ]
    return experiments

def train_deep_qlearning():
    try:
        environment = gym.make("LunarLander-v3")
    except Exception:
        environment = gym.make("LunarLander-v2")
    experiments = set_experiments()
    total_episodes_train = 5000
    total_episodes_test = 1000
    repetitions = 3
    pbar = tqdm(total=repetitions*len(experiments), desc='Optimización DQN', colour='green')
    for experiment in experiments:
        for i in range(repetitions):
            pbar.update(1)
            qlearning = QLearningDQN(environment=environment, params=experiment)
            print(30*'#')
            print('TRAIN ' + experiment['exp_name'])
            print(30*'#')
            #total_episodes = experiment['total_episodes']
            # TRAIN!
            qlearning.train(total_episodes=total_episodes_train)
            #qlearning.save_model(filename='mlpregressor_qtable_lunar_lander.pkl')
            qlearning.save_results(experiment_name=experiment['exp_name'],
                                   type_result='train')
            qlearning.reset_results()
            print(30 * '#')
            print('TEST ' + experiment['exp_name'])
            print(30 * '#')
            qlearning.test(total_episodes=total_episodes_test, save_results=True)
            qlearning.save_results(experiment_name=experiment['exp_name'],
                                   type_result='test')
            #environment.close()


if __name__ == "__main__":
    train_deep_qlearning()
