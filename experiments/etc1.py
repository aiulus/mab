# Implements Experiment 6.1 in 'Bandit Algorithms' by Tor Lattimore

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class ExploreThenCommit:
    def __init__(self, m: int, horizon: int):
        self.m = m
        self.horizon = horizon
        self.num_arms = 2
        self.counts = [0, 0]
        self.rewards = [[], []]
        self.committed_arm = None

    def select_arm(self, t: int):
        if t < self.m * self.num_arms:
            return t % self.num_arms
        if self.committed_arm is None:
            means = [np.mean(r) if r else 0.0 for r in self.rewards]
            self.committed_arm = int(np.argmax(means))
        return self.committed_arm

    def update(self, arm: int, reward: float):
        self.counts[arm] += 1
        self.rewards[arm].append(reward)


def run_etc_experiment(delta: float, m: int, n: int, num_trials: int):
    regrets = []

    for _ in range(num_trials):
        mu = [0.0, -delta]
        optimal_mean = max(mu)

        etc = ExploreThenCommit(m, n)
        cumulative_regret = 0.0

        for t in range(n):
            arm = etc.select_arm(t)
            reward = np.random.normal(mu[arm], 1.0)
            etc.update(arm, reward)
            regret = optimal_mean - mu[arm]
            cumulative_regret += regret

        regrets.append(cumulative_regret)

    return np.mean(regrets)


if __name__ == "__main__":
    n = 1000
    m = 10
    num_trials = 100_000
    deltas = np.linspace(0.01, 1.0, 20)
    regrets = []

    for delta in tqdm(deltas, desc="Running ETC experiments"):
        mean_regret = run_etc_experiment(delta, m, n, num_trials)
        regrets.append(mean_regret)

    plt.figure(figsize=(8, 5))
    plt.plot(deltas, regrets, marker='o', label=f"ETC (m={m}, n={n})")
    plt.xlabel("Suboptimality gap Δ")
    plt.ylabel("Expected Cumulative Regret")
    plt.title("Explore-Then-Commit: Regret vs Δ")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/plots/etc1.png")
    plt.show()
