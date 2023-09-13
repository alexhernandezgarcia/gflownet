import common
import pytest
import torch

from gflownet.envs.crystals.miller import MillerIndices


@pytest.fixture
def cubic():
    return MillerIndices(is_cubic=True)


@pytest.fixture
def nocubic():
    return MillerIndices(is_cubic=False)


@pytest.mark.parametrize(
    "state, state2oracle",
    [
        (
            [0, 0, 0],
            [-2.0, -2.0, -2.0],
        ),
        (
            [4, 4, 4],
            [2.0, 2.0, 2.0],
        ),
        (
            [2, 2, 2],
            [0.0, 0.0, 0.0],
        ),
        (
            [0, 2, 0],
            [-2.0, 0.0, -2.0],
        ),
        (
            [2, 1, 3],
            [0.0, -1.0, 1.0],
        ),
    ],
)
def test__state2oracle__cubic__returns_expected(cubic, state, state2oracle):
    env = cubic
    assert state2oracle == env.state2oracle(state)


@pytest.mark.parametrize(
    "state, state2oracle",
    [
        (
            [0, 0, 0, 0],
            [-2.0, -2.0, -2.0, -2.0],
        ),
        (
            [4, 4, 4, 4],
            [2.0, 2.0, 2.0, 2.0],
        ),
        (
            [2, 2, 2, 2],
            [0.0, 0.0, 0.0, 0.0],
        ),
        (
            [0, 2, 0, 2],
            [-2.0, 0.0, -2.0, 0.0],
        ),
        (
            [2, 1, 3, 0],
            [0.0, -1.0, 1.0, -2.0],
        ),
    ],
)
def test__state2oracle__nocubic__returns_expected(nocubic, state, state2oracle):
    env = nocubic
    assert state2oracle == env.state2oracle(state)


def test__all_env_common__cubic(cubic):
    print("\n\nCommon tests for cubic Miller indices\n")
    return common.test__all_env_common(cubic)


def test__all_env_common__nocubic(nocubic):
    print("\n\nCommon tests for hexagonal or rhombohedral Miller indices\n")
    return common.test__all_env_common(nocubic)
