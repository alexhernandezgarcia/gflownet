import common
import pytest

from gflownet.envs.grid import Grid


@pytest.fixture
def env():
    return Grid(n_dim=2, length=5)


def test__get_parents_step_get_mask__are_compatible(env):
    return common.test__get_parents_step_get_mask__are_compatible(env)


def test__gflownet_minimal_runs(env):
    return common.test__gflownet_minimal_runs(env)
