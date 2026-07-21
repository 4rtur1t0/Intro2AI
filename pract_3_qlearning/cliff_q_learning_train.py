from qlearning.qlearning import QLearning
import gymnasium as gym


def train_qlearning():
    environment = gym.make('CliffWalking-v1', render_mode='human')
    qlearning = QLearning(environment=environment)
    print('LET US LEARN NOW!')
    qlearning.train(total_episodes_train=300)
    qlearning.save_q_table(filename='qtable.npy')




if __name__ == "__main__":
    train_qlearning()
