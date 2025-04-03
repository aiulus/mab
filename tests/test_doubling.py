from algorithms.utils import DoublingTrickWrapper
from algorithms.etc import ExploreThenCommit


def test_doubling_wrapper_initializes():
    wrapper = DoublingTrickWrapper(
        bandit_class=ExploreThenCommit,
        bandit_kwargs={"n_arms": 2, "m": 2},
    )
    for _ in range(5):
        arm = wrapper.select_arm()
        wrapper.update(arm, reward=1.0)
    assert wrapper.current_t == 5
