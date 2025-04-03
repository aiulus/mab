import numpy as np
from bandits.base import Bandit


class GaussianBandit(Bandit):
    """
    Multi-armed bandit with Gaussian-distributed rewards.
    Each arm has a fixed mean and standard deviation (default σ=1).
    """

    def __init__(self, mu: list[float], sigma: float = 1.0, seed: int = None):
        self.mu = np.array(mu)
        self.sigma = sigma
        super().__init__(n_arms=len(mu), seed=seed)

    def pull(self, arm: int) -> float:
        """
        Sample a reward from a Gaussian distribution N(mu[arm], sigma^2).
        """
        reward = self.rng.normal(loc=self.mu[arm], scale=self.sigma)
        self.t += 1
        self.history.append((arm, reward))
        return reward

    def optimal_arm(self) -> int:
        """
        Return the index of the arm with the highest expected reward.
        """
        return int(np.argmax(self.mu))

    def expected_rewards(self) -> np.ndarray:
        """
        Return the expected rewards of all arms.
        """
        return self.mu
