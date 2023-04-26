import common
import numpy as np
import pytest
import torch

from gflownet.envs.ctorus import ContinuousTorus


@pytest.fixture
def env():
    return ContinuousTorus(n_dim=2, length_traj=3)


@pytest.mark.parametrize(
    "action_space",
    [
        [
            (0.0, 0.0),
            (np.inf, np.inf),
        ],
    ],
)
def test__get_action_space__returns_expected(env, action_space):
    assert set(action_space) == set(env.action_space)


def test__continuous_env_common(env):
    return common.test__continuous_env_common(env)
