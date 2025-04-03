from typing import Type
import numpy as np
from bandits.base import Bandit
from algorithms.base import BanditAlgorithm
from algorithms.etc import ExploreThenCommit
from bandits.gaussian import GaussianBandit
from bandits.bernoulli import BernoulliBandit
from bandits.linear import LinearBandit
import matplotlib.pyplot as plt
import os


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


def create_bandit(env_config: dict) -> Bandit:
    env_type = env_config["type"].lower()

    if env_type == "gaussian":
        return GaussianBandit(
            means=env_config["means"],
            std=env_config.get("std", 1.0),
            seed=env_config.get("seed")
        )
    elif env_type == "bernoulli":
        return BernoulliBandit(
            probs=env_config["probs"],
            seed=env_config.get("seed")
        )
    elif env_type == "linear":
        return LinearBandit(
            arms=env_config["arms"],
            theta=env_config["theta"],
            noise_std=env_config.get("noise_std", 0.1),
            seed=env_config.get("seed")
        )
    else:
        raise ValueError(f"Unknown bandit type: {env_type}")


def plot_regret(cum_regret: np.ndarray, label: str, save_path: str = None):
    plt.figure(figsize=(8, 5))
    plt.plot(cum_regret, label=label)
    plt.xlabel("Rounds")
    plt.ylabel("Cumulative Regret")
    plt.title("Bandit Algorithm Performance")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    plt.show()
