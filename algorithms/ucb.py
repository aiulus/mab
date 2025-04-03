import numpy as np
from .base import BanditAlgorithm


class UCB(BanditAlgorithm):
    """
    Upper Confidence Bound (UCB) algorithm for stochastic multi-armed bandits.
    Based on 1-subgaussian rewards and the optimism principle.

    Uses:
        UCB_i(t) = empirical_mean_i + sqrt(2 * log(1/δ) / n_i)
    """

    def __init__(self, n_arms: int, horizon: int, delta: float = None):
        super().__init__(n_arms)
        self.horizon = horizon
        self.delta = delta if delta is not None else 1.0 / (horizon ** 2)

        self.counts = np.zeros(n_arms, dtype=int)       # T_i(t)
        self.means = np.zeros(n_arms)                   # μ̂_i(t)
        self.ucbs = np.full(n_arms, np.inf)             # UCB_i(t)

    def select_arm(self) -> int:
        return int(np.argmax(self.ucbs))

    def update(self, arm: int, reward: float):
        self.counts[arm] += 1
        alpha = 1.0 / self.counts[arm]
        self.means[arm] += alpha * (reward - self.means[arm])
        self.t += 1

        # Update UCBs for all arms
        for i in range(self.n_arms):
            if self.counts[i] == 0:
                self.ucbs[i] = np.inf
            else:
                bonus = np.sqrt(2 * np.log(1.0 / self.delta) / self.counts[i])
                self.ucbs[i] = self.means[i] + bonus

    def name(self):
        return f"UCB(δ={self.delta:.2e})"
