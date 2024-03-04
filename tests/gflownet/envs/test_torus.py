import common
import numpy as np
import pytest
import torch

from gflownet.envs.torus import Torus


@pytest.fixture
def env():
    return Torus(n_dim=3, n_angles=5)


@pytest.fixture
def env_extended_action_space_2d():
    return Torus(
        n_dim=2,
        n_angles=5,
        max_increment=2,
        max_dim_per_action=-1,
    )


@pytest.fixture
def env_extended_action_space_3d():
    return Torus(
        n_dim=3,
        n_angles=5,
        max_increment=2,
        max_dim_per_action=2,
    )


@pytest.mark.parametrize(
    "action_space",
    [
        [
            (-2, -2),
            (-2, -1),
            (-2, 0),
            (-2, 1),
            (-2, 2),
            (-1, -2),
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (-1, 2),
            (0, -2),
            (0, -1),
            (0, 0),
            (0, 1),
            (0, 2),
            (1, -2),
            (1, -1),
            (1, 0),
            (1, 1),
            (1, 2),
            (2, -2),
            (2, -1),
            (2, 0),
            (2, 1),
            (2, 2),
            (3, 3),
        ],
    ],
)
def test__get_action_space__returns_expected(
    env_extended_action_space_2d, action_space
):
    assert set(action_space) == set(env_extended_action_space_2d.action_space)


class TestTorusBasic(common.BaseTestsDiscrete):
    @pytest.fixture(autouse=True)
    def setup(self, env):
        self.env = env
        self.repeats = {
            "test__reset__state_is_source": 10,
        }
        self.n_states = {}  # TODO: Populate.


class TestTorusExtendedActionSpace3D(common.BaseTestsDiscrete):
    @pytest.fixture(autouse=True)
    def setup(self, env_extended_action_space_3d):
        self.env = env_extended_action_space_3d
        self.repeats = {
            "test__reset__state_is_source": 10,
        }
        self.n_states = {}  # TODO: Populate.
