import numpy as np
import matplotlib.pyplot as plt
from bandits.gaussian import GaussianBandit
from algorithms.ucb import UCB
from experiments.runner import ExperimentRunner
from evaluation.metrics import compute_mean_regret_over_runs, compute_confidence_interval

if __name__ == "__main__":
    delta = 0.05
    n_rounds = 1000
    n_trials = 100
    means = [0.0, -0.5]
    bandit_envs = [GaussianBandit(mu=means, seed=i) for i in range(n_trials)]

    all_regrets = []

    for env in bandit_envs:
        algo = UCB(n_arms=2, horizon=n_rounds, delta=delta)
        runner = ExperimentRunner(env, algo, n_rounds)
        runner.run()
        all_regrets.append(runner.cum_regret)

    all_regrets = np.array(all_regrets)
    mean = compute_mean_regret_over_runs(all_regrets)
    lower, upper = compute_confidence_interval(all_regrets)

    plt.plot(mean, label="UCB")
    plt.fill_between(np.arange(n_rounds), lower, upper, alpha=0.2)
    plt.title("Cumulative Regret of UCB")
    plt.xlabel("Rounds")
    plt.ylabel("Cumulative Regret")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/plots/ucb_regret.png")
    plt.show()
