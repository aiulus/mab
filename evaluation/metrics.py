# evaluation/metrics.py

import numpy as np


def compute_regret(rewards: np.ndarray, optimal_reward: float) -> np.ndarray:
    """
    Compute instantaneous regret: the gap between optimal and received reward.
    """
    return optimal_reward - rewards


def compute_cumulative_regret(regret: np.ndarray) -> np.ndarray:
    """
    Compute cumulative regret over time.
    """
    return np.cumsum(regret)


def compute_mean_regret_over_runs(all_regrets: np.ndarray) -> np.ndarray:
    """
    Compute average cumulative regret across multiple experiment runs.
    Shape of all_regrets: (n_runs, n_rounds)
    """
    return all_regrets.mean(axis=0)


def compute_confidence_interval(all_regrets: np.ndarray, confidence: float = 0.95) -> tuple:
    """
    Compute confidence interval for cumulative regret.
    """
    import scipy.stats as stats

    mean = all_regrets.mean(axis=0)
    sem = stats.sem(all_regrets, axis=0)
    margin = sem * stats.t.ppf((1 + confidence) / 2., df=all_regrets.shape[0] - 1)
    return mean - margin, mean + margin
