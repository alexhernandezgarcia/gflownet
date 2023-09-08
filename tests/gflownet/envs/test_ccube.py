import common
import numpy as np
import pytest
import torch

from gflownet.envs.cube import ContinuousCube


@pytest.fixture
def cube1d():
    return ContinuousCube(n_dim=1, n_comp=3, min_incr=0.1, max_val=1.0)


@pytest.fixture
def cube2d():
    return ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1, max_val=1.0)


@pytest.mark.parametrize(
    "action_space",
    [
        [
            (0.0, 0.0),
            (-1.0, -1.0),
            (np.inf, np.inf),
        ],
    ],
)
@pytest.mark.skip(reason="skip while developping other tests")
def test__get_action_space__returns_expected(env, action_space):
    assert set(action_space) == set(env.action_space)


@pytest.mark.parametrize("env", ["cube1d", "cube2d"])
def test__get_policy_output__fixed_as_expected(env, request):
    env = request.getfixturevalue(env)
    policy_output = env.fixed_policy_output
    params = env.fixed_distr_params
    policy_output__as_expected(env, policy_output, params)


@pytest.mark.parametrize("env", ["cube1d", "cube2d"])
def test__get_policy_output__random_as_expected(env, request):
    env = request.getfixturevalue(env)
    policy_output = env.random_policy_output
    params = env.random_distr_params
    policy_output__as_expected(env, policy_output, params)


def policy_output__as_expected(env, policy_output, params):
    assert torch.all(
        env._get_policy_betas_weights(policy_output) == params["beta_weights"]
    )
    assert torch.all(env._get_policy_betas_alpha(policy_output) == params["beta_alpha"])
    assert torch.all(env._get_policy_betas_beta(policy_output) == params["beta_beta"])
    assert torch.all(
        env._get_policy_bw_zero_increment_logits(policy_output)
        == params["bernoulli_bw_zero_incr_logits"]
    )
    assert torch.all(
        env._get_policy_eos_logit(policy_output) == params["bernoulli_eos_logit"]
    )
    assert torch.all(
        env._get_policy_source_logit(policy_output) == params["bernoulli_source_logit"]
    )


@pytest.mark.parametrize("env", ["cube1d", "cube2d"])
def test__mask_forward__returns_all_true_if_done(env, request):
    env = request.getfixturevalue(env)
    # Sample states
    states = env.get_uniform_terminating_states(100)
    # Iterate over state and test
    for state in states:
        env.set_state(state, done=True)
        mask = env.get_mask_invalid_actions_forward()
        assert all(mask)


@pytest.mark.parametrize("env", ["cube1d", "cube2d"])
def test__mask_backward__returns_all_true_except_eos_if_done(env, request):
    env = request.getfixturevalue(env)
    # Sample states
    states = env.get_uniform_terminating_states(100)
    # Iterate over state and test
    for state in states:
        env.set_state(state, done=True)
        mask = env.get_mask_invalid_actions_backward()
        assert all(mask[:-1])
        assert mask[-1] is False


@pytest.mark.parametrize(
    "state, mask_expected",
    [
        (
            [0.0],
            [True, False, True],
        ),
        (
            [0.5],
            [True, True, False],
        ),
        (
            [0.90],
            [True, True, False],
        ),
        (
            [0.95],
            [False, True, False],
        ),
    ],
)
def test__mask_forward__1d__returns_expected(cube1d, state, mask_expected):
    env = cube1d
    mask = env.get_mask_invalid_actions_forward(state)
    assert mask == mask_expected


@pytest.mark.parametrize(
    "state, mask_expected",
    [
        (
            [0.0, 0.0],
            [True, True, False, True],
        ),
        (
            [0.5, 0.5],
            [True, True, True, False],
        ),
        (
            [0.90, 0.5],
            [True, True, True, False],
        ),
        (
            [0.95, 0.5],
            [False, True, True, False],
        ),
        (
            [0.5, 0.90],
            [True, True, True, False],
        ),
        (
            [0.5, 0.95],
            [True, False, True, False],
        ),
    ],
)
def test__mask_forward__2d__returns_expected(cube2d, state, mask_expected):
    env = cube2d
    mask = env.get_mask_invalid_actions_forward(state)
    assert mask == mask_expected


@pytest.mark.parametrize(
    "state, mask_expected",
    [
        (
            [0.0],
            [True, True, True],
        ),
        (
            [0.1],
            [True, True, True],
        ),
        (
            [0.05],
            [True, False, True],
        ),
        (
            [0.5],
            [True, True, True],
        ),
        (
            [0.90],
            [True, True, True],
        ),
        (
            [0.95],
            [True, True, True],
        ),
    ],
)
def test__mask_backward__1d__returns_expected(cube1d, state, mask_expected):
    env = cube1d
    mask = env.get_mask_invalid_actions_backward(state)
    assert mask == mask_expected


@pytest.mark.parametrize(
    "state, mask_expected",
    [
        (
            [0.0, 0.0],
            [True, True, True, True],
        ),
        (
            [0.5, 0.5],
            [True, True, True, True],
        ),
        (
            [0.05, 0.5],
            [True, True, False, True],
        ),
        (
            [0.5, 0.05],
            [True, True, False, True],
        ),
        (
            [0.05, 0.05],
            [True, True, False, True],
        ),
        (
            [0.90, 0.5],
            [True, True, True, True],
        ),
        (
            [0.5, 0.90],
            [True, True, True, True],
        ),
        (
            [0.95, 0.5],
            [False, True, True, True],
        ),
        (
            [0.5, 0.95],
            [True, False, True, True],
        ),
        (
            [0.95, 0.95],
            [False, False, True, True],
        ),
    ],
)
def test__mask_backward__2d__returns_expected(cube2d, state, mask_expected):
    env = cube2d
    mask = env.get_mask_invalid_actions_backward(state)
    assert mask == mask_expected


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
@pytest.mark.skip(reason="skip while developping other tests")
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
@pytest.mark.skip(reason="skip while developping other tests")
def test__statetorch2policy_returns_expected(env, states, expected):
    assert torch.equal(
        env.statetorch2policy(torch.tensor(states)), torch.tensor(expected)
    )


@pytest.mark.parametrize(
    "state, expected",
    [
        (
            [0.0, 0.0],
            [True, False, False],
        ),
        (
            [0.1, 0.1],
            [False, True, False],
        ),
        (
            [1.0, 0.0],
            [False, True, False],
        ),
        (
            [1.1, 0.0],
            [True, True, False],
        ),
        (
            [0.1, 1.1],
            [True, True, False],
        ),
    ],
)
@pytest.mark.skip(reason="skip while developping other tests")
def test__get_mask_invalid_actions_forward__returns_expected(env, state, expected):
    assert env.get_mask_invalid_actions_forward(state) == expected, print(
        state, expected, env.get_mask_invalid_actions_forward(state)
    )


@pytest.mark.skip(reason="skip while developping other tests")
def test__continuous_env_common(env):
    return common.test__continuous_env_common(env)
