import numpy as np
from bandits.base import Bandit


class LinearBandit(Bandit):
    """
    Contextual linear bandit: rewards are linear in a given context.
    r = ⟨theta, x⟩ + noise
    """

    def __init__(self, arms: list[np.ndarray], theta: np.ndarray, noise_std: float = 0.1, seed: int = None):
        """
        :param arms: List of context vectors (1 per arm)
        :param theta: True parameter vector (same dimension as context)
        :param noise_std: Std deviation of additive Gaussian noise
        """
        self.arms = np.array(arms)  # shape: (n_arms, d)
        self.theta = np.array(theta)
        self.noise_std = noise_std
        super().__init__(n_arms=len(arms), seed=seed)

    def pull(self, arm: int) -> float:
        x = self.arms[arm]
        mean_reward = np.dot(self.theta, x)
        reward = mean_reward + self.rng.normal(0, self.noise_std)
        self.t += 1
        self.history.append((arm, reward))
        return reward

    def optimal_arm(self) -> int:
        return int(np.argmax(self.arms @ self.theta))

    def expected_rewards(self) -> np.ndarray:
        return self.arms @ self.theta
