"""
    A solution for QLearning.
    Es poden resoldre diferents escenes d'aprenentatge amb aquest mateix disseny.
"""
import numpy as np
import time
import random

class QLearning():
    def __init__(self, environment):
        self.env = environment
        # Initialize Q-table with zeros (500 states x 6 actions)
        self.state_size = self.env.observation_space.n
        self.action_size = self.env.action_space.n
        # importante, la tabla Q se inicializa a ceros por defecto
        # se puede cargar desde un fichero con el método load_q_table
        # se puede guardar con el método save_q_table
        self.q_table = np.zeros((self.state_size, self.action_size))
        # Hyperparameters
        self.learning_rate = 1.0  # alpha
        self.discount_rate = 1.0  # gamma
        self.epsilon = 1.0  # Exploration rate
        self.max_epsilon = 1.0
        self.min_epsilon = 0.01
        self.decay_rate = 0.995  # Exponential decay rate for exploration
        # para guardar resultados
        self.results = []
        # during training, test each testing_episodes
        self.testing_episodes = 10

    def train(self, total_episodes):
        self.results = []
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

    def test(self, total_episodes):
        print('Test started')
        # test loop
        for episode in range(total_episodes):
            print("Episode: ", episode)
            state, info = self.env.reset()
            srt = 0
            while True:
                # Greedy action selection
                action = np.argmax(self.q_table[state])  # Exploit
                # Take action, observe new state and reward
                next_state, reward, terminated, truncated, info = self.env.step(action)
                srt += reward
                time.sleep(.1)
                if terminated or truncated:
                    self.results.append(srt)
                    break
                # Move to the next state
                state = next_state
        print("Test finished!")
        self.env.close()

    def create_random_q_table(self):
        self.q_table = np.random.rand(self.state_size, self.action_size)

    def read_q_table(self, filename):
        with open(filename, 'rb') as f:
            self.q_table = np.load(f)

    def save_q_table(self, filename):
        with open(filename, 'wb') as f:
            np.save(f, self.q_table)




