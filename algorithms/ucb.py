import numpy as np
from algorithms.base import BanditAlgorithm


class UCB1(BanditAlgorithm):
    def __init__(self, n_arms: int):
        super().__init__(n_arms)
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)

    def reset(self):
        super().reset()
        self.counts[:] = 0
        self.values[:] = 0

    def select_arm(self) -> int:
        self.t += 1
        if 0 in self.counts:
            return int(np.argmin(self.counts))  # pull untried arm

        ucb_values = self.values + np.sqrt(2 * np.log(self.t) / self.counts)
        return int(np.argmax(ucb_values))

    def update(self, arm: int, reward: float):
        self.counts[arm] += 1
        n = self.counts[arm]
        value = self.values[arm]
        self.values[arm] = ((n - 1) / n) * value + (1 / n) * reward
