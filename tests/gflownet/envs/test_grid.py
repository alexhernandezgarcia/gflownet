import common
import pytest
import torch

from gflownet.envs.grid import Grid
from gflownet.utils.common import tfloat


@pytest.fixture
def env():
    return Grid(n_dim=3, length=5, cell_min=-1.0, cell_max=1.0)


@pytest.fixture
def env_extended_action_space_2d():
    return Grid(
        n_dim=2,
        length=5,
        max_increment=2,
        max_dim_per_action=-1,
        cell_min=-1.0,
        cell_max=1.0,
    )


@pytest.fixture
def env_extended_action_space_3d():
    return Grid(
        n_dim=3,
        length=5,
        max_increment=2,
        max_dim_per_action=3,
        cell_min=-1.0,
        cell_max=1.0,
    )


@pytest.fixture
def env_default():
    return Grid()


@pytest.fixture
def config_path():
    return "../../../config/env/grid.yaml"


@pytest.mark.parametrize(
    "state, state2proxy",
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
def test__state2proxy__returns_expected(env, state, state2proxy):
    assert torch.equal(
        tfloat(state2proxy, device=env.device, float_type=env.float),
        env.state2proxy(state),
    )


@pytest.mark.parametrize(
    "states, states2proxy",
    [
        (
            [[0, 0, 0], [4, 4, 4], [1, 2, 3], [4, 0, 1]],
            [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], [-0.5, 0.0, 0.5], [1.0, -1.0, -0.5]],
        ),
    ],
)
def test__states2proxy__returns_expected(env, states, states2proxy):
    assert torch.equal(torch.Tensor(states2proxy), env.states2proxy(states))


@pytest.mark.parametrize(
    "action_space",
    [
        [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2)],
    ],
)
def test__get_action_space__returns_expected(
    env_extended_action_space_2d, action_space
):
    assert set(action_space) == set(env_extended_action_space_2d.action_space)


def test__all_env_common__standard(env_extended_action_space_3d):
    print("\n\nCommon tests for 5x5 Grid with extended action space\n")
    return common.test__all_env_common(env_extended_action_space_3d)


def test__all_env_common__extended(env):
    print("\n\nCommon tests for 5x5 Grid with standard action space\n")
    return common.test__all_env_common(env)
