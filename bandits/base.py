# bandits/base.py

from abc import ABC, abstractmethod
import numpy as np


class Bandit(ABC):
    """
    Abstract base class for a multi-armed bandit environment.
    """

    def __init__(self, n_arms: int, seed: int = None):
        self.n_arms = n_arms
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        """Reset the environment for a new run."""
        self.t = 0
        self.history = []

    @abstractmethod
    def pull(self, arm: int) -> float:
        """Pull a specific arm and return the reward."""
        pass

    @abstractmethod
    def optimal_arm(self) -> int:
        """Return the index of the optimal arm."""
        pass

    @abstractmethod
    def expected_rewards(self) -> np.ndarray:
        """Return the expected reward for each arm."""
        pass
