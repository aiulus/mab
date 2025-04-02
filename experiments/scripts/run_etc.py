import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from experiments.runner import run_etc

if __name__ == "__main__":
    n = 1000
    m = 10
    trials = 100_000
    deltas = np.linspace(0.01, 1.0, 20)

    regrets = []
    for delta in tqdm(deltas, desc="Running ETC Experiments"):
        avg_regret = run_etc(delta, m=m, horizon=n, num_trials=trials)
        regrets.append(avg_regret)

    plt.figure(figsize=(8, 5))
    plt.plot(deltas, regrets, marker='o', label=f"ETC (m={m})")
    plt.xlabel("Suboptimality gap Δ")
    plt.ylabel("Expected Cumulative Regret")
    plt.title("Explore-Then-Commit: Regret vs Δ")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    os.makedirs("outputs/plots", exist_ok=True)
    plt.savefig("outputs/plots/etc1.png")
    plt.show()
