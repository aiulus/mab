from typing import Type
import numpy as np
from bandits.base import Bandit
from algorithms.base import BanditAlgorithm
from algorithms.etc import ExploreThenCommit


class ExperimentRunner:
    """
    Runs a bandit algorithm in a bandit environment and records regret.
    """

    def __init__(
        self,
        bandit: Bandit,
        algorithm: BanditAlgorithm,
        n_rounds: int,
        seed: int = None,
    ):
        self.bandit = bandit
        self.algorithm = algorithm
        self.n_rounds = n_rounds
        self.rng = np.random.default_rng(seed)

        self.rewards = np.zeros(n_rounds)
        self.actions = np.zeros(n_rounds, dtype=int)
        self.regret = np.zeros(n_rounds)
        self.cum_regret = np.zeros(n_rounds)

    def run(self):
        self.bandit.reset()
        self.algorithm.reset()

        optimal_reward = self.bandit.expected_rewards()[self.bandit.optimal_arm()]

        for t in range(self.n_rounds):
            arm = self.algorithm.select_arm()
            reward = self.bandit.pull(arm)
            self.algorithm.update(arm, reward)

            self.actions[t] = arm
            self.rewards[t] = reward
            self.regret[t] = optimal_reward - reward
            self.cum_regret[t] = self.regret[: t + 1].sum()

    def summary(self) -> dict:
        return {
            "mean_reward": self.rewards.mean(),
            "total_reward": self.rewards.sum(),
            "total_regret": self.cum_regret[-1],
            "regret": self.regret,
            "cum_regret": self.cum_regret,
            "actions": self.actions,
        }


def run_etc(delta: float, m: int, horizon: int, num_trials: int):
    regrets = []

    for _ in range(num_trials):
        means = [0.0, -delta]
        optimal = max(means)

        algo = ExploreThenCommit(n_arms=2, horizon=horizon, m=m)
        regret = 0.0

        for t in range(horizon):
            arm = algo.select_arm(t)
            reward = np.random.normal(means[arm], 1.0)
            algo.update(arm, reward)
            regret += optimal - means[arm]

        regrets.append(regret)

    return np.mean(regrets)
