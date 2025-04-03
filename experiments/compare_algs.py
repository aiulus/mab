import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from bandits.gaussian import GaussianBandit
from algorithms.etc import ExploreThenCommit
from algorithms.e_greedy import EpsilonGreedy
from algorithms.ucb import UCB
from experiments.runner import ExperimentRunner
from evaluation.metrics import (
    compute_mean_regret_over_runs,
    compute_confidence_interval,
)
from evaluation.plots import plot_multiple_regret_curves
from utils.io import ensure_output_dirs

ensure_output_dirs()


def run_trials(algo_class, algo_kwargs, bandit_config, n_rounds=1000, n_trials=100, seed=0):
    all_regrets = []

    for i in tqdm(range(n_trials), desc=f"Running {algo_class.__name__}"):
        bandit = GaussianBandit(mu=bandit_config, sigma=1.0, seed=i + seed)
        algo = algo_class(n_arms=len(bandit_config), **algo_kwargs)
        runner = ExperimentRunner(bandit, algo, n_rounds=n_rounds)
        runner.run()
        all_regrets.append(runner.cum_regret)

    all_regrets = np.array(all_regrets)
    mean = compute_mean_regret_over_runs(all_regrets)
    lower, upper = compute_confidence_interval(all_regrets)
    return {"mean": mean, "lower": lower, "upper": upper}


if __name__ == "__main__":
    n_rounds = 1000
    n_trials = 100
    bandit_config = [0.0, -0.5]

    results = {
        "ETC (m=25)": run_trials(ExploreThenCommit, {"m": 25, "horizon": n_rounds}, bandit_config, n_rounds, n_trials),
        "ε-greedy (ε=0.1)": run_trials(EpsilonGreedy, {"epsilon_value": 0.1, "epsilon_schedule": "constant"},
                                       bandit_config, n_rounds, n_trials),
        "UCB": run_trials(UCB, {"horizon": n_rounds}, bandit_config, n_rounds, n_trials)
    }

    plot_multiple_regret_curves(results, title="Algorithm Comparison (Gaussian Bandit)", ylabel="Cumulative Regret")
    plt.savefig("outputs/plots/benchmark_gaussian.png")
    plt.show()
