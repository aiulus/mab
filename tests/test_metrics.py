import numpy as np
from evaluation.metrics import compute_regret, compute_cumulative_regret


def test_regret_computation():
    rewards = np.array([1, 0.5, 0.0])
    optimal = 1.0
    regret = compute_regret(rewards, optimal)
    assert np.allclose(regret, [0.0, 0.5, 1.0])
    cum = compute_cumulative_regret(regret)
    assert np.allclose(cum, [0.0, 0.5, 1.5])
