# evaluation/plotting.py

import matplotlib.pyplot as plt
import numpy as np


def plot_regret_curve(
    mean_regret: np.ndarray,
    lower_ci: np.ndarray = None,
    upper_ci: np.ndarray = None,
    label: str = None,
    color: str = None
):
    """
    Plot a single regret curve with optional confidence interval.
    """
    x = np.arange(len(mean_regret))
    plt.plot(x, mean_regret, label=label, color=color, linewidth=2)

    if lower_ci is not None and upper_ci is not None:
        plt.fill_between(x, lower_ci, upper_ci, color=color, alpha=0.2)


def plot_multiple_regret_curves(results: dict, title="Cumulative Regret", ylabel="Regret", xlabel="Rounds"):
    """
    Plot multiple regret curves on the same plot.

    :param results: Dict mapping algorithm names to dicts with keys: 'mean', 'lower', 'upper'
    """
    plt.figure(figsize=(10, 6))

    for i, (alg_name, metrics) in enumerate(results.items()):
        color = f"C{i}"  # Use default matplotlib color cycle
        plot_regret_curve(
            mean_regret=metrics["mean"],
            lower_ci=metrics.get("lower"),
            upper_ci=metrics.get("upper"),
            label=alg_name,
            color=color
        )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
