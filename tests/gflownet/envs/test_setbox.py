import warnings

import common
import numpy as np
import pytest
import torch
from torch import Tensor

from gflownet.envs.setbox import SetBox
from gflownet.utils.common import copy, tbool, tfloat


@pytest.fixture
def env_1d_max1():
    return SetBox(
        n_dim=1,
        max_elements_per_subenv=1,
    )


@pytest.fixture
def env_2d_max2():
    return SetBox(
        n_dim=2,
        max_elements_per_subenv=2,
    )


@pytest.fixture
def env_2d_max3():
    return SetBox(
        n_dim=2,
        max_elements_per_subenv=3,
    )


@pytest.mark.parametrize(
    "env",
    [
        "env_1d_max1",
        "env_2d_max2",
        "env_2d_max3",
    ],
)
def test__environment__initializes_properly(env, request):
    env = request.getfixturevalue(env)
    assert True


@pytest.mark.parametrize(
    "env, state, dones",
    [
        (
            "env_1d_max1",
            [1, [1, 1], [[-1, 0, [1, 1], [0, 1]], {0: [0.5431], 1: [0]}]],
            [True, False],
        ),
        (
            "env_1d_max1",
            [1, [0, 1], [[-1, 0, [1, 1], [1, -1]], {0: [1]}]],
            [True, True],
        ),
    ],
)
def test__set_state__sets_state_and_dones(env, state, dones, request):
    env = request.getfixturevalue(env)
    if all(dones):
        env.set_state(state, done=True)
    else:
        env.set_state(state, done=False)

    # Check global state
    assert env.equal(env.state, state)

    # Check states of subenvs
    for stage, subenv in env.subenvs.items():
        assert env.equal(subenv.state, env._get_substate(state, stage))

    # Check dones
    for subenv, done in zip(env.subenvs.values(), dones):
        assert subenv.done == done


@pytest.mark.parametrize(
    "env, state, mask_exp",
    [
        # Source state of second subenv (Set)
        # The only valid action is EOS of the conditioning grid to transition backwards
        (
            "env_1d_max1",
            [1, [1, 1], [[-1, 0, [0, 0], [0, 1]], {0: [-1], 1: [0]}]],
            # fmt: off
            [
                True, False, # ACTIVE SUBENV (Stack)
                True, True, False, # MASK (Conditioning Grid)
                False, False, False, # PADDING
            ]
            # fmt: on
        ),
        # The (only) state in the Set is done (but the Stack is not)
        # The active env in the Set is -1.
        # The only valid action (backwards) is activating subenv 0
        (
            "env_1d_max1",
            [1, [0, 1], [[-1, 0, [1, 1], [1, -1]], {0: [0]}]],
            # fmt: off
            [
                False, True, # ACTIVE SUBENV (Stack)
                False, False, # ACTIVE SUBENV (Set, no active subenv)
                False, True, True, # MASK (Set)
                False, # PADDING
            ]
            # fmt: on
        ),
        (
            "env_1d_max1",
            [1, [0, 0], [[-1, 0, [1, 1], [1, -1]], {0: [0]}]],
            # fmt: off
            [
                False, True, # ACTIVE SUBENV (Stack)
                False, False, # ACTIVE SUBENV (Set, no active subenv)
                False, True, True, # MASK (Set)
                False, # PADDING
            ]
            # fmt: on
        ),
    ],
)
def test__get_mask_invalid_actions_backward__returns_expected(
    env, state, mask_exp, request
):
    env = request.getfixturevalue(env)
    # Passing state
    mask = env.get_mask_invalid_actions_backward(state, done=False)
    assert mask == mask_exp
    # State from env
    env.set_state(state, done=False)
    mask = env.get_mask_invalid_actions_backward()
    assert mask == mask_exp


@pytest.mark.parametrize(
    "env, state_from, action, state_next_exp, valid_exp",
    [
        (
            "env_1d_max1",
            [1, [0, 1], [[-1, 0, [1, 1], [1, -1]], {0: [1]}]],
            (1, -1, -1, -1),
            [1, [0, 1], [[-1, 0, [1, 1], [1, -1]], {0: [1]}]],
            True,
        ),
        (
            "env_1d_max1",
            [1, [1, 1], [[-1, 0, [1, 1], [0, 1]], {0: [0.4321], 1: [1]}]],
            (1, -1, -1, -1),
            [1, [1, 1], [[-1, 0, [1, 1], [0, 1]], {0: [0.4321], 1: [1]}]],
            True,
        ),
    ],
)
def test__step_backwards_from_global_done__works_as_expected(
    env, state_from, action, state_next_exp, valid_exp, request
):
    env = request.getfixturevalue(env)
    env.set_state(state_from, done=True)

    # Check init state
    assert env.equal(env.state, state_from)

    # Perform step
    warnings.filterwarnings("ignore")
    state_next, action_done, valid = env.step_backwards(action)

    # Check end state
    assert env.equal(env.state, state_next)
    assert env.equal(env.state, state_next_exp)

    # Check action and valid
    assert action_done == action
    assert valid == valid_exp, (state_from, action)


@pytest.mark.repeat(5)
@pytest.mark.parametrize(
    "env",
    [
        "env_1d_max1",
        "env_2d_max2",
        "env_2d_max3",
    ],
)
def test__step_random__does_not_crash_from_source(env, request):
    env = request.getfixturevalue(env)
    env.reset()
    state_next, action, valid = env.step_random()
    assert True


@pytest.mark.repeat(5)
@pytest.mark.parametrize(
    "env",
    [
        "env_1d_max1",
        "env_2d_max2",
        "env_2d_max3",
    ],
)
def test__step_random__does_not_crash_and_reaches_done(env, request):
    env = request.getfixturevalue(env)
    env.reset()
    states = [copy(env.state)]
    actions = []
    while not env.done:
        state_next, action, valid = env.step_random()
        if valid:
            states.append(copy(state_next))
            actions.append(action)
        else:
            warnings.warn("IMPORTANT: Found invalid action!")
    assert True


@pytest.mark.repeat(10)
@pytest.mark.parametrize(
    "env",
    [
        "env_1d_max1",
        "env_2d_max2",
        "env_2d_max3",
    ],
)
def test__trajectory_random__does_not_crash_and_reaches_done(env, request):
    env = request.getfixturevalue(env)
    env.reset()
    env.trajectory_random()
    assert env.done


@pytest.mark.repeat(5)
@pytest.mark.parametrize(
    "env",
    [
        "env_1d_max1",
        "env_2d_max2",
        "env_2d_max3",
    ],
)
def test__step_random_backwards__does_not_crash_and_reaches_source(env, request):
    env = request.getfixturevalue(env)
    env.reset()
    # Sample a full trajectory forward
    env.trajectory_random()
    states = [copy(env.state)]
    actions = []
    while not env.is_source():
        state_next, action, valid = env.step_random(backward=True)
        if valid:
            states.append(copy(state_next))
            actions.append(action)
        else:
            warnings.warn("IMPORTANT: Found invalid action!")
    assert True


class TestSet1DMax1(common.BaseTestsContinuous):
    """Common tests for setbox in 1D and with maximum 1 environment of each kind."""

    @pytest.fixture(autouse=True)
    def setup(self, env_1d_max1):
        self.env = env_1d_max1
        self.repeats = {
            "test__reset__state_is_source": 10,
            "test__forward_actions_have_nonzero_backward_prob": 10,
            "test__backward_actions_have_nonzero_forward_prob": 10,
            "test__trajectories_are_reversible": 10,
            "test__step_random__does_not_sample_invalid_actions_forward": 10,
            "test__step_random__does_not_sample_invalid_actions_backward": 10,
            "test__get_mask__is_consistent_regardless_of_inputs": 10,
            "test__get_valid_actions__is_consistent_regardless_of_inputs": 10,
            "test__get_parents_step_get_mask__are_compatible": 10,
            "test__sample_backwards_reaches_source": 10,
            "test__state2readable__is_reversible": 20,
            "test__gflownet_minimal_runs": 1,
        }
        self.n_states = {
            "test__backward_actions_have_nonzero_forward_prob": 3,
            "test__sample_backwards_reaches_source": 3,
            "test__get_logprobs__all_finite_in_random_forward_transitions": 10,
            "test__get_logprobs__all_finite_in_random_backward_transitions": 10,
        }
        self.batch_size = {
            "test__sample_actions__get_logprobs__batched_forward_trajectories": 10,
            "test__sample_actions__get_logprobs__batched_backward_trajectories": 10,
            "test__get_logprobs__all_finite_in_accumulated_forward_trajectories": 10,
            "test__gflownet_minimal_runs": 5,
        }


class TestSet2DMax3(common.BaseTestsContinuous):
    """Common tests for setbox in 1D and with maximum 1 environment of each kind."""

    @pytest.fixture(autouse=True)
    def setup(self, env_2d_max3):
        self.env = env_2d_max3
        self.repeats = {
            "test__reset__state_is_source": 10,
            "test__forward_actions_have_nonzero_backward_prob": 10,
            "test__backward_actions_have_nonzero_forward_prob": 10,
            "test__trajectories_are_reversible": 10,
            "test__step_random__does_not_sample_invalid_actions_forward": 10,
            "test__step_random__does_not_sample_invalid_actions_backward": 10,
            "test__get_mask__is_consistent_regardless_of_inputs": 10,
            "test__get_valid_actions__is_consistent_regardless_of_inputs": 10,
            "test__get_parents_step_get_mask__are_compatible": 10,
            "test__sample_backwards_reaches_source": 10,
            "test__state2readable__is_reversible": 20,
            "test__gflownet_minimal_runs": 1,
        }
        self.n_states = {
            "test__backward_actions_have_nonzero_forward_prob": 3,
            "test__sample_backwards_reaches_source": 3,
            "test__get_logprobs__all_finite_in_random_forward_transitions": 10,
            "test__get_logprobs__all_finite_in_random_backward_transitions": 10,
        }
        self.batch_size = {
            "test__sample_actions__get_logprobs__batched_forward_trajectories": 10,
            "test__sample_actions__get_logprobs__batched_backward_trajectories": 10,
            "test__get_logprobs__all_finite_in_accumulated_forward_trajectories": 10,
            "test__gflownet_minimal_runs": 5,
        }
