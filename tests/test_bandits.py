import numpy as np
from bandits.bernoulli import BernoulliBandit
from bandits.gaussian import GaussianBandit
from bandits.linear import LinearBandit


def test_bernoulli_bandit():
    bandit = BernoulliBandit(probs=[0.2, 0.8], seed=42)
    rewards = [bandit.pull(0) for _ in range(100)]
    assert set(rewards).issubset({0, 1})
    assert bandit.optimal_arm() == 1
    np.testing.assert_array_almost_equal(bandit.expected_rewards(), [0.2, 0.8])


def test_gaussian_bandit():
    bandit = GaussianBandit(mu=[0.0, 1.0], seed=42)
    rewards = [bandit.pull(1) for _ in range(100)]
    assert all(isinstance(r, float) for r in rewards)
    assert bandit.optimal_arm() == 1


def test_linear_bandit():
    arms = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    theta = np.array([0.5, 1.0])
    bandit = LinearBandit(arms=arms, theta=theta, seed=42)
    assert bandit.optimal_arm() == 1
    np.testing.assert_array_almost_equal(bandit.expected_rewards(), [0.5, 1.0])
