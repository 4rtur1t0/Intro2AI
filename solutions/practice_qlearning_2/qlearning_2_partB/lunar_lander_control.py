from qlearning_DQN.qlearning_DQN import QLearningDQN
import gymnasium as gym
import numpy as np
import scipy.linalg
import gymnasium as gym


def compute_lqr_gain(g=10.0, m=1.0, d=0.5, I=0.1):
    # State: [x, y, theta, vx, vy, omega]
    A = np.array([
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, -g, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]
    ])

    B = np.array([
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [1.0 / m, 0],
        [0, d / I]
    ])

    Q = np.diag([15.0, 20.0, 15.0, 5.0, 5.0, 5.0])
    R = np.diag([0.1, 0.1])

    P = scipy.linalg.solve_continuous_are(A, B, Q, R)
    return np.linalg.inv(R) @ B.T @ P


def map_continuous_to_discrete(u_main, u_side, main_thresh=0.05, side_thresh=0.1):
    """Discretizes continuous LQR demands into Gym actions {0, 1, 2, 3}."""
    # Prioritize rotational stability
    if u_main > main_thresh:
        return 2  # Fire main engine
    if u_side > side_thresh:
        return 1  # Fire left engine (rotates clockwise / pushes right)
    elif u_side < -side_thresh:
        return 3  # Fire right engine (rotates counter-clockwise / pushes left)

    return 0  # Do nothing


def run_discrete_lqr():
    try:
        env = gym.make("LunarLander-v3", render_mode="human")
    except Exception:
        env = gym.make("LunarLander-v2", render_mode="human")

    obs, _ = env.reset()
    K = compute_lqr_gain()

    hover_gravity_offset = 0.5  # Feedforward hover baseline
    done = False

    while not done:
        # Gymnasium observation array: [x, y, vx, vy, theta, omega, leg1, leg2]
        x, y, vx, vy, theta, omega = obs[:6]
        state = np.array([x, y, theta, vx, vy, omega])

        # Calculate continuous LQR force demands
        u_delta = -K @ state
        u_main = u_delta[0] + hover_gravity_offset
        u_side = u_delta[1]

        # Select discrete action
        action = map_continuous_to_discrete(u_main, u_side)

        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    env.close()


if __name__ == "__main__":
    run_discrete_lqr()