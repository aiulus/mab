# bandits/bernoulli.py

import numpy as np
from bandits.base import Bandit


class BernoulliBandit(Bandit):
    """
    Multi-armed bandit with Bernoulli rewards.
    Each arm returns 1 with probability p, 0 otherwise.
    """

    def __init__(self, probs: list[float], seed: int = None):
        super().__init__(n_arms=len(probs), seed=seed)
        self.probs = np.array(probs)

    def pull(self, arm: int) -> float:
        self.t += 1
        reward = self.rng.binomial(1, self.probs[arm])
        self.history.append((arm, reward))
        return reward

    def optimal_arm(self) -> int:
        return int(np.argmax(self.probs))

    def expected_rewards(self) -> np.ndarray:
        return self.probs.copy()
