# The Explore-Then-Commit Algorithm

import numpy as np
from algorithms.base import BanditAlgorithm


class ExploreThenCommit(BanditAlgorithm):
    """
    Explore-Then-Commit algorithm (ETC) for K-armed bandits.

    Explores each arm `m` times, then commits to the one with the highest empirical mean.
    """

    def __init__(self, n_arms: int, horizon: int, m: int):
        super().__init__(n_arms)
        self.horizon = horizon
        self.m = m
        self.total_explore_rounds = m * n_arms
        self.means = np.zeros(n_arms)
        self.counts = np.zeros(n_arms, dtype=int)
        self.rewards = [[] for _ in range(n_arms)]
        self.committed_arm = None

    def select_arm(self, t: int) -> int:
        if t < self.total_explore_rounds:
            return t % self.n_arms
        else:
            if self.committed_arm is None:
                # Compute empirical means and commit
                for i in range(self.n_arms):
                    self.means[i] = np.mean(self.rewards[i])
                self.committed_arm = int(np.argmax(self.means))
            return self.committed_arm

    def update(self, arm: int, reward: float):
        self.counts[arm] += 1
        self.rewards[arm].append(reward)

    def name(self):
        return f"ETC (m={self.m})"
