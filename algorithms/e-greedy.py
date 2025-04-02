import numpy as np
from algorithms.base import BanditAlgorithm


class EpsilonGreedy(BanditAlgorithm):
    def __init__(self, num_arms: int, epsilon_schedule='constant', epsilon_value=0.1):
        self.num_arms = num_arms
        self.counts = np.zeros(num_arms, dtype=int)
        self.values = np.zeros(num_arms)
        self.t = 0
        self.epsilon_schedule = epsilon_schedule
        self.epsilon_value = epsilon_value

    def _epsilon_t(self):
        if self.epsilon_schedule == 'constant':
            return self.epsilon_value
        elif self.epsilon_schedule == 'decay':
            return min(1.0, self.epsilon_value / max(1, self.t))
        else:
            raise ValueError(f"Unknown epsilon schedule: {self.epsilon_schedule}")

    def select_arm(self) -> int:
        self.t += 1
        if self.t <= self.num_arms:
            return self.t - 1  # initialize by pulling each arm once
        if np.random.rand() < self._epsilon_t():
            return np.random.randint(self.num_arms)
        return int(np.argmax(self.values))

    def update(self, arm: int, reward: float):
        self.counts[arm] += 1
        alpha = 1 / self.counts[arm]
        self.values[arm] += alpha * (reward - self.values[arm])
