import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from algorithms.etc import ExploreThenCommit
from bandits.gaussian import GaussianBandit
from experiments.runner import ExperimentRunner
from evaluation.metrics import compute_mean_regret_over_runs, compute_confidence_interval
from evaluation.plots import plot_multiple_regret_curves
from utils.io import ensure_output_dirs

ensure_output_dirs()


def run_etc(delta: float, m: int, horizon: int, num_trials: int):
    regrets = []
    for i in tqdm(range(num_trials), desc=f"ETC (Δ={delta:.2f})"):
        bandit = GaussianBandit(mu=[0.0, -delta], seed=i)
        algo = ExploreThenCommit(n_arms=2, m=m, horizon=horizon)
        runner = ExperimentRunner(bandit, algo, n_rounds=horizon)
        runner.run()
        regrets.append(runner.cum_regret[-1])
    return np.mean(regrets)


if __name__ == "__main__":
    n = 1000
    m = 25
    num_trials = 500
    deltas = np.linspace(0.01, 1.0, 20)
    regrets = []

    for delta in deltas:
        mean_regret = run_etc(delta, m, n, num_trials)
        regrets.append(mean_regret)

    plt.figure(figsize=(8, 5))
    plt.plot(deltas, regrets, marker='o', label=f"ETC (m={m}, n={n})")
    plt.xlabel("Suboptimality gap Δ")
    plt.ylabel("Expected Cumulative Regret")
    plt.title("Explore-Then-Commit: Regret vs Δ")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/plots/etc_refactored.png")
    plt.show()
