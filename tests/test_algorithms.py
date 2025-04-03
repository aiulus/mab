import numpy as np
from algorithms.etc import ExploreThenCommit
from algorithms.e_greedy import EpsilonGreedy


def test_etc_behavior():
    etc = ExploreThenCommit(n_arms=2, m=5, horizon=100)
    for t in range(11):  # Explore phase is 10 rounds
        arm = etc.select_arm(t)
        etc.update(arm, reward=float(arm))  # deterministic
    assert etc.committed_arm in [0, 1]


def test_eps_greedy():
    eg = EpsilonGreedy(num_arms=3, epsilon_schedule="constant", epsilon_value=0.1)
    for _ in range(10):
        arm = eg.select_arm()
        eg.update(arm, reward=np.random.rand())
    assert eg.counts.sum() == 10
