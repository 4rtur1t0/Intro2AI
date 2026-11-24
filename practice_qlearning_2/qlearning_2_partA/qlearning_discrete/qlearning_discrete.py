"""
    A solution for QLearning.
    Es tracta de tabular Q learning on es discretitza un estat continu.
    Es poden resoldre diferents escenes d'aprenentatge amb aquest mateix disseny.
"""
import numpy as np
import time
from collections import defaultdict
import pickle
from libAI.epsilon_greedy import EpsilonGreedyGeom
from tqdm import tqdm

# Non-uniform bin boundaries: tighter resolution around 0 for fine control
X_BINS = np.array([-0.4, -0.15, -0.05, 0.05, 0.15, 0.4])
Y_BINS = np.array([0.05, 0.2, 0.5, 0.9])
VX_BINS = np.array([-0.5, -0.2, -0.05, 0.05, 0.2, 0.5])
VY_BINS = np.array([-0.8, -0.3, -0.1, 0.0, 0.2])
ANGLE_BINS = np.array([-0.4, -0.1, -0.02, 0.02, 0.1, 0.4])
ANG_VEL_BINS = np.array([-0.6, -0.2, 0.2, 0.6])


class QLearningD():
    """
    Q learning con discretización
    """
    def __init__(self, environment, params={}):
        self.env = environment
        self.action_size = self.env.action_space.n
        # importante, la tabla Q la constituye un diccionario.
        # Devuelve ceros si se solicita una clave que no existe.
        self.q_table = defaultdict(lambda: np.zeros(self.action_size))
        # Hyperparameters
        self.alpha = params.get('alpha', 0.5)
        self.gamma = params.get('gamma', 1.0)
        self.epsilon_max = params.get('epsilon_max', 1.0)
        self.epsilon_min = params.get('epsilon_min', 0.01)
        self.epsilon_percentage = params.get('epsilon_percentage', 0.25)  # ratio de episodios para llegar a epsilon_min
        self.training_tests = params.get('training_tests', (50, 10))

    def train(self, total_episodes):
        epsilon_greedy = EpsilonGreedyGeom(epsilon_max=self.epsilon_max, epsilon_min=self.epsilon_min,
                                           total_episodes=total_episodes, percentage_target=self.epsilon_percentage)
        print("Training started!\n\n")
        epsilons = []
        rewards_train = []
        rewards_inline_test = []
        avg_window = 500
        # Training loop
        pbar = tqdm(range(total_episodes), desc='Discrete Q-learning train')
        for episode in pbar:
            state, info = self.env.reset()
            state_d = self.discretize_state(state)
            while True:
                # Epsilon-greedy action selection
                if epsilon_greedy.random_action():
                    action = self.env.action_space.sample()  # Explore
                else:
                    action = np.argmax(self.q_table[state_d])  # Exploit
                # Take action, observe new state and reward
                next_state, reward, terminated, truncated, info = self.env.step(action)
                next_state_d = self.discretize_state(next_state)
                # Actualiza la tabla
                done = terminated or truncated
                self.update_q_table(state_d, action, next_state_d, reward, done)
                if done:
                    break
                # Move to next state
                state_d = next_state_d
            # Reduce epsilon (less exploration, more exploitation as time goes on)
            epsilon_greedy.step()
            # Afegeix informació a la barra
            pbar.set_postfix(mean_train_reward=-50.0,
                             epsilon=epsilon_greedy.epsilon,
                             mean_test_reward=-50.0)
        return rewards_train, rewards_inline_test, epsilons

    def update_q_table(self, state_d, action, next_state_d, reward, done):
        if not done:
            # Update Q-table using the Bellman Equation
            self.q_table[state_d][action] = self.q_table[state_d][action] + \
                                            self.alpha * (reward
                                                          + self.gamma * np.max(self.q_table[next_state_d]) -
                                                            self.q_table[state_d][action])
        else:
            # Update Q-table using the Bellman Equation, with Q(s',a)=0
            self.q_table[state_d][action] = (self.q_table[state_d][action] + \
                                             self.alpha * (reward - self.q_table[state_d][action]))

    def test(self, total_episodes):
        results_test = []
        for episode in range(total_episodes):
            state, info = self.env.reset()
            state_d = self.discretize_state(state)
            while True:
                # Greedy action selection
                action = np.argmax(self.q_table[state_d])  # Exploit
                # Take action, observe new state and reward
                next_state, reward, terminated, truncated, info = self.env.step(action)
                next_state_d = self.discretize_state(next_state)
                if self.env.render_mode == 'human':
                    time.sleep(.1)
                if terminated or truncated:
                    break
                # Move to the next state
                state_d = next_state_d
        return results_test

    def create_random_q_table(self):
        """
        In case you want to start with a random q table... and not zeros
        :return:
        """
        # Distribución normal centrada en 0 (media=0, std=1)
        self.q_table = defaultdict(lambda: np.random.randn(self.action_size))

    def read_model(self, filename):
        with open(filename, "rb") as f:
            raw_q_table = pickle.load(f)
        self.q_table = defaultdict(lambda: np.zeros(self.action_size), raw_q_table)

    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(dict(self.q_table), f)

    def discretize_state(self, state):
        """Maps the 8D continuous state into a discrete tuple of bin indices."""
        x, y, vx, vy, angle, ang_vel, leg1, leg2 = state
        return (
            int(np.digitize(x, X_BINS)),
            int(np.digitize(y, Y_BINS)),
            int(np.digitize(vx, VX_BINS)),
            int(np.digitize(vy, VY_BINS)),
            int(np.digitize(angle, ANGLE_BINS)),
            int(np.digitize(ang_vel, ANG_VEL_BINS)),
            int(leg1),
            int(leg2),
        )