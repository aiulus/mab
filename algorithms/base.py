from abc import ABC, abstractmethod


class BanditAlgorithm(ABC):
    """
    Abstract base class for a bandit algorithm.
    """

    def __init__(self, n_arms: int):
        self.n_arms = n_arms
        self.t = 0
        self.reset()

    def reset(self):
        self.t = 0

    @abstractmethod
    def select_arm(self) -> int:
        """Select which arm to pull next."""
        pass

    @abstractmethod
    def update(self, arm: int, reward: float):
        """Update internal state with observed reward."""
        pass
