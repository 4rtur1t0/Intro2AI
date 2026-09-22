from libAI.epsilon_greedy import EpsilonGreedyGeom, EpsilonGreedyLinear, EpsilonGreedyInverseTime, EpsilonGreedyExp
import matplotlib.pyplot as plt

def test_epsilon_greedy():
    total_episodes = 1000
    epsilon_greedy_geom = EpsilonGreedyGeom(epsilon_max=1.0, epsilon_min=0.01,
                                            total_episodes=total_episodes,
                                            percentage_target=0.25)
    epsilon_greedy_exp = EpsilonGreedyExp(epsilon_max=1.0, epsilon_min=0.01,
                                            total_episodes=total_episodes,
                                            percentage_target=0.25)
    epsilon_greedy_linear = EpsilonGreedyLinear(epsilon_max=1.0, epsilon_min=0.01,
                                            total_episodes=total_episodes,
                                            percentage_target=0.25)
    epsilon_greedy_inv_time = EpsilonGreedyInverseTime(epsilon_max=1.0, epsilon_min=0.01,
                                            total_episodes=total_episodes,
                                            percentage_target=0.25)
    for episode in range(total_episodes):
        epsilon_greedy_geom.step()
        epsilon_greedy_exp.step()
        epsilon_greedy_linear.step()
        epsilon_greedy_inv_time.step()
    epsilon_greedy_geom.plot('Epsilon-greedy geometric')
    epsilon_greedy_exp.plot('Epsilon-greedy exponential')
    epsilon_greedy_linear.plot('Epsilon-greedy linear')
    epsilon_greedy_inv_time.plot('Epsilon-greedy inverse-time')
    plt.show()


if __name__ == "__main__":
    test_epsilon_greedy()
