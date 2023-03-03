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


def test__state_conversions_are_reversible(env):
    return common.test__state_conversions_are_reversible(env)


def test__get_parents_step_get_mask__are_compatible(env):
    return common.test__get_parents_step_get_mask__are_compatible(env)


def test__get_parents__returns_no_parents_in_initial_state(env):
    return common.test__get_parents__returns_no_parents_in_initial_state(env)


def test__get_parents__returns_same_state_and_eos_if_done(env):
    return common.test__get_parents__returns_same_state_and_eos_if_done(env)


def test__sample_actions__get_logprobs__return_valid_actions_and_logprobs(env):
    return common.test__sample_actions__get_logprobs__return_valid_actions_and_logprobs(
        env
    )


def test__sample_backwards_reaches_source(env):
    return common.test__sample_backwards_reaches_source(env)


# def test__default_config_equals_default_args(env_default, config_path):
#     return common.test__default_config_equals_default_args(env_default, config_path)


def test__gflownet_minimal_runs(env):
    return common.test__gflownet_minimal_runs(env)
