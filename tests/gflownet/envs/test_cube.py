import common
import numpy as np
import pytest
import torch

from gflownet.envs.cube import HybridCube


@pytest.fixture
def env():
    return HybridCube(n_dim=2, n_comp=3)


@pytest.mark.parametrize(
    "action_space",
    [
        [
            (0, 0.0),
            (1, 0.0),
            (2, 0.0),
        ],
    ],
)
def test__get_action_space__returns_expected(env, action_space):
    assert set(action_space) == set(env.action_space)


def test__get_policy_output__returns_expected(env):
    assert env.policy_output_dim == env.n_dim * env.n_comp * 3 + env.n_dim + 1
    fixed_policy_output = env.fixed_policy_output
    random_policy_output = env.random_policy_output
    assert torch.all(fixed_policy_output[: env.n_dim + 1] == 1)
    assert torch.all(random_policy_output[: env.n_dim + 1] == 1)
    assert torch.all(fixed_policy_output[env.n_dim + 1 :: 3] == 1)
    assert torch.all(
        fixed_policy_output[env.n_dim + 2 :: 3] == env.fixed_distr_params["beta_alpha"]
    )
    assert torch.all(
        fixed_policy_output[env.n_dim + 3 :: 3] == env.fixed_distr_params["beta_beta"]
    )
    assert torch.all(random_policy_output[env.n_dim + 1 :: 3] == 1)
    assert torch.all(
        random_policy_output[env.n_dim + 2 :: 3]
        == env.random_distr_params["beta_alpha"]
    )
    assert torch.all(
        random_policy_output[env.n_dim + 3 :: 3] == env.random_distr_params["beta_beta"]
    )


@pytest.mark.parametrize(
    "state, expected",
    [
        (
            [0.0, 0.0],
            [0.0, 0.0],
        ),
        (
            [1.0, 1.0],
            [1.0, 1.0],
        ),
        (
            [1.1, 1.00001],
            [1.0, 1.0],
        ),
        (
            [-0.1, 1.00001],
            [0.0, 1.0],
        ),
        (
            [0.1, 0.21],
            [0.1, 0.21],
        ),
    ],
)
def test__state2policy_returns_expected(env, state, expected):
    assert env.state2policy(state) == expected


@pytest.mark.parametrize(
    "states, expected",
    [
        (
            [[0.0, 0.0], [1.0, 1.0], [1.1, 1.00001], [-0.1, 1.00001], [0.1, 0.21]],
            [[0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [0.0, 1.0], [0.1, 0.21]],
        ),
    ],
)
def test__statebatch_torch2policy_returns_expected(env, states, expected):
    assert np.equal(env.statebatch2policy(states), np.array(expected)).all()
    assert torch.equal(
        env.statetorch2policy(torch.tensor(states)), torch.tensor(expected)
    )


# def test__continuous_env_common(env):
#     return common.test__continuous_env_common(env)
