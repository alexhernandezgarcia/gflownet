import common
import pytest

from gflownet.envs.grid import Grid


@pytest.fixture
def env():
    return Grid(n_dim=2, length=5)

@pytest.fixture
def env_default():
    return Grid()

@pytest.fixture
def config_path():
    return "../../../config/env/grid.yaml"


def test__get_parents_step_get_mask__are_compatible(env):
    return common.test__get_parents_step_get_mask__are_compatible(env)


# def test__default_config_equals_default_args(env_default, config_path):
#     return common.test__default_config_equals_default_args(env_default, config_path) 


def test__gflownet_minimal_runs(env):
    return common.test__gflownet_minimal_runs(env)
