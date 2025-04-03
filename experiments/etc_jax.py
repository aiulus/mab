import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


# === JAX-based ETC simulation ===
def run_etc_jax(delta: float, m: int, n: int, num_trials: int, key):
    mu = jnp.array([0.0, -delta])
    optimal = jnp.max(mu)

    explore_len = 2 * m
    exploit_len = n - explore_len

    keys = jax.random.split(key, num_trials)

    def simulate_trial(k):
        k1, k2 = jax.random.split(k)

        # === Step 1: Exploration ===
        rewards = jax.random.normal(k1, shape=(2, m)) + mu[:, None]
        means = jnp.mean(rewards, axis=1)
        committed_arm = jnp.argmax(means)

        # === Step 2: Exploitation ===
        exploit_rewards = jax.random.normal(k2, shape=(exploit_len,)) + mu[committed_arm]

        # Total reward & regret
        explore_reward = jnp.sum(rewards)
        total_reward = explore_reward + jnp.sum(exploit_rewards)
        regret = n * optimal - total_reward
        return regret

    regrets = jax.vmap(simulate_trial)(keys)
    return jnp.mean(regrets)


# === Regret Bound from Theorem 6.1 / Equation (6.6) ===
def regret_bound(delta, n):
    if delta == 0:
        return 0.0
    log_term = max(0, jnp.log(n * delta ** 2 / 4))
    return min(n * delta, delta + (4 / delta) * (1 + log_term))


# === Main Execution ===
if __name__ == "__main__":
    n = 1000
    m = 25
    num_trials = 500
    deltas = jnp.linspace(0.01, 1.0, 20)

    regrets = []
    bounds = []
    key = jax.random.PRNGKey(0)

    for i, delta in enumerate(tqdm(deltas, desc="JAX ETC + Bound")):
        trial_key = jax.random.fold_in(key, i)
        regret = run_etc_jax(float(delta), m, n, num_trials, trial_key)
        regrets.append(float(regret))
        bounds.append(float(regret_bound(float(delta), n)))

    # === Plotting ===
    plt.figure(figsize=(8, 5))
    plt.plot(deltas, regrets, marker='o', label=f"ETC (m={m})")
    plt.plot(deltas, bounds, linestyle='--', color='red', label="Theoretical Bound")
    plt.xlabel("Suboptimality gap Δ")
    plt.ylabel("Expected Cumulative Regret")
    plt.title("Explore-Then-Commit: JAX vs Theoretical Bound")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    os.makedirs("outputs/plots", exist_ok=True)
    plt.savefig("outputs/plots/etc_jax_vs_bound.png")
    plt.show()
