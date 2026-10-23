import common
import numpy as np
import pytest
import torch
from torch.distributions import Bernoulli, Beta

from gflownet.envs.cube_cond_dim import CubeCondDim
from gflownet.utils.common import tbool, tfloat


@pytest.fixture
def cube_cond_dim():
    """
    Default CubeCondDim
        - dimensions_tr: 2
        - max_dimensions: 3
    """
    return CubeCondDim()


@pytest.fixture
def env_tr2_max2():
    """
    2 dimensions for training and maximum 2 dimensions.
    """
    return CubeCondDim(
        dimensions_tr=[1, 2], max_dimension=2, cube_kwargs={"min_incr": 0.1}
    )


@pytest.fixture
def env_tr2_max3_cond2():
    """
    2 dimensions for training, and maximum 3 dimensions, fixed condition 2.
    """
    return CubeCondDim(
        dimensions_tr=[1, 2],
        max_dimension=3,
        cube_kwargs={"min_incr": 0.1},
        condition=[2],
    )


@pytest.mark.parametrize(
    "env",
    [
        "cube_cond_dim",
        "env_tr2_max2",
        "env_tr2_max3_cond2",
    ],
)
def test__environment__initializes_properly(env, request):
    env = request.getfixturevalue(env)
    assert True


@pytest.mark.parametrize(
    "env, n_dim",
    [
        ("cube_cond_dim", 3),
        ("env_tr2_max2", 2),
        ("env_tr2_max3_cond2", 3),
    ],
)
def test__cube_n_dim__as_expected(env, n_dim, request):
    env = request.getfixturevalue(env)
    assert env.cube.n_dim == n_dim


def test__env_with_fixed_condition_sets_condition_correctly(env_tr2_max3_cond2):
    env = env_tr2_max3_cond2
    assert env.condition_env.state == [2]
    env.reset()
    assert env.condition_env.state == [2]


@pytest.mark.parametrize(
    "action_space",
    [
        [
            # Constant EOS
            (0, 0, 0, 0, 0),
            # 3D Cube actions
            (1, 0.0, 0.0, 0.0, 0.0),
            (1, 0.0, 0.0, 0.0, 1.0),
            (1, np.inf, np.inf, np.inf, np.inf),
        ],
    ],
)
def test__get_action_space__returns_expected(cube_cond_dim, action_space):
    env = cube_cond_dim
    assert action_space == env.action_space


@pytest.mark.repeat(5)
@pytest.mark.parametrize(
    "env",
    [
        "cube_cond_dim",
        "env_tr2_max2",
        "env_tr2_max3_cond2",
    ],
)
def test__step_random__does_not_crash_from_source(env, request):
    env = request.getfixturevalue(env)
    env.reset()
    state_next, action, valid = env.step_random()
    assert True


@pytest.mark.repeat(5)
def test__env_with_fixed_condition_sets_ignored_dims_correctly(env_tr2_max3_cond2):
    env = env_tr2_max3_cond2
    env.reset()
    state_next, action, valid = env.step_random()
    assert sum(env.cube.ignored_dims) == (env.cube.n_dim - 2)


@pytest.mark.repeat(5)
@pytest.mark.parametrize(
    "env",
    [
        "cube_cond_dim",
        "env_tr2_max2",
        "env_tr2_max3_cond2",
    ],
)
def test__multiple_random_steps__do_not_crash_from_source(env, request):
    env = request.getfixturevalue(env)
    env.reset()
    while not env.done:
        state_next, action, valid = env.step_random()
    assert env.done


@pytest.mark.repeat(5)
@pytest.mark.parametrize(
    "env",
    [
        "cube_cond_dim",
        "env_tr2_max2",
        "env_tr2_max3_cond2",
    ],
)
def test__trajectory_random__reaches_done(env, request):
    env = request.getfixturevalue(env)
    env.reset()
    state, actions = env.trajectory_random()
    assert env.done
