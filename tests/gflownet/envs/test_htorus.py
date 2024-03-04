import common
import numpy as np
import pytest
import torch

from gflownet.envs.htorus import HybridTorus


@pytest.fixture
def env():
    return HybridTorus(n_dim=2, length_traj=3)


@pytest.mark.parametrize(
    "action_space",
    [
        [
            (0, 0),
            (1, 0),
            (2, 0),
        ]
    ],
)
def test__get_action_space__returns_expected(env, action_space):
    assert set(action_space) == set(env.action_space)


@pytest.mark.skip(reason="skip while the environment remains outdated")
class TestHybridTorus(common.BaseTestsDiscrete):
    @pytest.fixture(autouse=True)
    def setup(self, env):
        self.env = env
        self.repeats = {
            "test__reset__state_is_source": 10,
        }
        self.n_states = {}  # TODO: Populate.
