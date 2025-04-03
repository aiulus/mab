from bandits.bernoulli import BernoulliBandit
from algorithms.etc import ExploreThenCommit
from experiments.runner import ExperimentRunner


def test_runner_basic():
    bandit = BernoulliBandit(probs=[0.1, 0.9], seed=123)
    algo = ExploreThenCommit(n_arms=2, m=2, horizon=20)
    runner = ExperimentRunner(bandit, algo, n_rounds=20)
    runner.run()
    summary = runner.summary()
    assert "total_regret" in summary
    assert len(summary["actions"]) == 20
