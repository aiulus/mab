import numpy as np


class DoublingTrickWrapper:
    """
    Wraps a horizon-dependent bandit algorithm using the doubling trick
    to create an anytime version.
    """

    def __init__(self, bandit_class, bandit_kwargs=None, doubling_sequence=None):
        """
        Parameters:
        - bandit_class: Callable class reference for the bandit algorithm (e.g., ExploreThenCommit)
        - bandit_kwargs: dict of parameters required to initialize the bandit
        - doubling_sequence: list or generator of increasing horizons (defaults to powers of 2)
        """
        self.bandit_class = bandit_class
        self.bandit_kwargs = bandit_kwargs or {}
        self.doubling_sequence = doubling_sequence or self._default_doubling_sequence()
        self.current_phase = 0
        self.current_t = 0
        self.phase_start_time = 0

        self._start_new_phase()

    def _default_doubling_sequence(self):
        i = 0
        while True:
            yield 2 ** i
            i += 1

    def _start_new_phase(self):
        horizon = next(self.doubling_sequence)
        self.phase_horizon = horizon
        self.phase_start_time = self.current_t
        self.phase_algorithm = self.bandit_class(horizon=horizon, **self.bandit_kwargs)

    def select_arm(self):
        if self.current_t - self.phase_start_time >= self.phase_horizon:
            self._start_new_phase()
        return self.phase_algorithm.select_arm(self.current_t)

    def update(self, arm: int, reward: float):
        self.phase_algorithm.update(arm, reward)
        self.current_t += 1
