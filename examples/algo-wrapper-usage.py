from algorithms.utils import DoublingTrickWrapper
from algorithms.etc import ExploreThenCommit
from bandits.gaussian import GaussianBandit  # Example subclass of Bandit

# Environment: standard 2-arm Gaussian bandit
bandit = GaussianBandit(mu=[0.0, -0.5], sigma=1.0)

# Agent: Doubling-trick-wrapped ETC
agent = DoublingTrickWrapper(
    bandit_class=ExploreThenCommit,
    bandit_kwargs={"m": 5}  # ETC's per-arm exploration count
)

# Simulation loop
n_steps = 500
for t in range(n_steps):
    arm = agent.select_arm()
    reward = bandit.pull(arm)
    agent.update(arm, reward)
