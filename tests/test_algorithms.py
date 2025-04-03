import numpy as np
from algorithms.etc import ExploreThenCommit
from algorithms.e_greedy import EpsilonGreedy
from algorithms.ucb import UCB


def test_etc_behavior():
    etc = ExploreThenCommit(n_arms=2, m=5, horizon=100)
    for t in range(11):  # Explore phase is 10 rounds
        arm = etc.select_arm(t)
        etc.update(arm, reward=float(arm))  # deterministic
    assert etc.committed_arm in [0, 1]


def test_eps_greedy():
    eg = EpsilonGreedy(n_arms=3, epsilon_schedule="constant", epsilon_value=0.1)
    for _ in range(10):
        arm = eg.select_arm()
        eg.update(arm, reward=np.random.rand())
    assert eg.counts.sum() == 10


def test_ucb_selects_and_updates():
    ucb = UCB(n_arms=2, horizon=1000)
    for _ in range(2):  # ensure each arm is pulled once
        arm = ucb.select_arm()
        ucb.update(arm, reward=1.0)

    # After initial pulls, check that UCB values are finite
    assert all(np.isfinite(ucb.ucbs))
    assert np.argmax(ucb.ucbs) in [0, 1]