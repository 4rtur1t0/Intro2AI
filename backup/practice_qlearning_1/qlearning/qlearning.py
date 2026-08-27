"""
    A solution for QLearning.
    Es poden resoldre diferents escenes d'aprenentatge amb aquest mateix disseny.
"""
import numpy as np
import time
import random

from libAI.epsilon_greedy import EpsilonGreedy


class QLearning():
    def __init__(self, environment, params):
        self.env = environment
        self.state_size = self.env.observation_space.n
        self.action_size = self.env.action_space.n
        # importante, la tabla Q se inicializa a ceros por defecto
        # se puede cargar desde un fichero con el método load_q_table
        # se puede guardar con el método save_q_table
        self.q_table = np.zeros((self.state_size, self.action_size))
        # Hyperparameters
        self.learning_rate = params.get('alpha', 1.0)  # alpha
        self.discount_rate = params.get('gamma', 1.0)  # gamma
        #self.epsilon = 1.0  # Exploration rate
        self.epsilon_max = params.get('epsilon_max', 1.0)
        self.epsilon_min = params.get('epsilon_min', 0.01)
        self.percentage_target = params.get('percentage_target', 0.25)  # Ntarget for exploration
        # para guardar resultados
        self.results = []
        #self.testing_episodes = 10
        # during training, test each 10 training episodes. Perform 5 tests only
        self.training_tests = (10, 5)

    def train(self, total_episodes):
        self.results = []
        epsilon_greedy = EpsilonGreedy(epsilon_max=self.epsilon_max,
                                       epsilon_min=self.epsilon_min,
                                       total_episodes=total_episodes,
                                       percentage_target=self.percentage_target)
        print("Training started!\n\n")
        # Training loop
        for episode in range(total_episodes):
            """
            EJERCICIO: SE DEBE COMPLETAR EL CÓDIGO POR EL ESTUDIANTE
            """

        print("Training finished! Your Q-table is optimized.")
        self.env.close()
        return self.q_table


    def update_q_table(self, state, action, next_state, reward):
        # Update Q-table using the Bellman Equation
        """
        ACTIVIDAD: SE DEBE PROGRAMAR LA REGLA DE ACTUALIZACION DE LA TABLA Q
        """
        #self.q_table[state, action] = ...

    def test(self, total_episodes, save_results=False):
        print('Test started')
        recent_rewards = []
        for episode in range(total_episodes):
            #print("Episode: ", episode)
            state, info = self.env.reset()
            total_reward = 0
            while True:
                # Greedy action selection
                action = np.argmax(self.q_table[state])  # Exploit
                # Take action, observe new state and reward
                next_state, reward, terminated, truncated, info = self.env.step(action)
                total_reward += reward
                if self.env.render_mode == 'human':
                    time.sleep(.1)
                if terminated or truncated:
                    break
                # Move to the next state
                state = next_state
            recent_rewards.append(total_reward)
            if save_results:
                self.results.append([episode, total_reward, 0])
            print(f"Episode {episode:4d} | Total reward: {total_reward:6.1f}", end='\r')
            if episode % 5 == 0:
                print()
        #print("Test finished!")
        self.env.close()

    def create_random_q_table(self):
        self.q_table = np.random.rand(self.state_size, self.action_size)

    def read_q_table(self, filename):
        with open(filename, 'rb') as f:
            self.q_table = np.load(f)

    def save_q_table(self, filename):
        with open(filename, 'wb') as f:
            np.save(f, self.q_table)




