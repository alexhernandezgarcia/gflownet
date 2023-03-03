import common
import pytest
import torch

from gflownet.envs.grid import Grid


@pytest.fixture
def env():
    return Grid(n_dim=3, length=5, cell_min=-1.0, cell_max=1.0)


@pytest.fixture
def env_default():
    return Grid()


@pytest.fixture
def config_path():
    return "../../../config/env/grid.yaml"


@pytest.mark.parametrize(
    "state, state2oracle",
    [
        (
            [0, 0, 0],
            [-1.0, -1.0, -1.0],
        ),
        (
            [4, 4, 4],
            [1.0, 1.0, 1.0],
        ),
        (
            [1, 2, 3],
            [-0.5, 0.0, 0.5],
        ),
        (
            [4, 0, 1],
            [1.0, -1.0, -0.5],
        ),
    ],
)
def test__state2oracle__returns_expected(env, state, state2oracle):
    assert state2oracle == env.state2oracle(state)


@pytest.mark.parametrize(
    "states, statebatch2oracle",
    [
        (
            [[0, 0, 0], [4, 4, 4], [1, 2, 3], [4, 0, 1]],
            [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], [-0.5, 0.0, 0.5], [1.0, -1.0, -0.5]],
        ),
    ],
)
def test__statebatch2oracle__returns_expected(env, states, statebatch2oracle):
    assert torch.equal(torch.Tensor(statebatch2oracle), env.statebatch2oracle(states))


def test__all_env_common(env):
    return common.test__all_env_common(env)
