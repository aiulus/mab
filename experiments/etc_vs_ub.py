import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# === Parameters ===
n = 1000
m = 25
num_trials = 500
deltas = np.linspace(0.01, 1.0, 20)


# === ETC Simulation ===
def run_etc(delta: float, m: int, horizon: int, num_trials: int):
    regrets = []
    for _ in range(num_trials):
        mu = [0.0, -delta]
        optimal = max(mu)

        rewards = [[], []]
        committed_arm = None
        regret = 0.0

        for t in range(horizon):
            if t < m * 2:
                arm = t % 2
            else:
                if committed_arm is None:
                    avg = [np.mean(r) for r in rewards]
                    committed_arm = int(np.argmax(avg))
                arm = committed_arm

            reward = np.random.normal(mu[arm], 1.0)
            rewards[arm].append(reward)
            regret += optimal - mu[arm]

        regrets.append(regret)

    return np.mean(regrets)


# === Regret Bound (Theorem 6.1 Simplified) ===
def regret_bound(delta, n):
    if delta == 0:
        return 0.0
    max_term = max(0, np.log(n * delta ** 2 / 4))
    return min(n * delta, delta + (4 / delta) * (1 + max_term))


# === Run Experiments ===
regrets = []
bounds = []

for delta in tqdm(deltas, desc="Running ETC and computing bounds"):
    reg = run_etc(delta, m, n, num_trials)
    regrets.append(reg)
    bounds.append(regret_bound(delta, n))

# === Plotting ===
plt.figure(figsize=(8, 5))
plt.plot(deltas, regrets, marker='o', label=f"ETC Empirical (m={m})")
plt.plot(deltas, bounds, linestyle='--', color='red', label="Theoretical Bound")
plt.xlabel("Suboptimality gap Δ")
plt.ylabel("Cumulative Regret")
plt.title("ETC: Empirical Regret vs Theoretical Upper Bound")
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save plot
os.makedirs("outputs/plots", exist_ok=True)
plt.savefig("outputs/plots/etc_vs_bound.png")
plt.show()
