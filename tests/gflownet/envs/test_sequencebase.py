import common
import pytest
import torch

from gflownet.envs.sequences.base import SequenceBase
from gflownet.utils.common import tlong


@pytest.fixture
def env():
    return SequenceBase(
        tokens=(-2, -1, 0, 1, 2), pad_token=10, max_length=5, device="cpu"
    )


@pytest.fixture
def env_no_min_length():
    return SequenceBase(
        tokens=(-2, -1, 0, 1, 2), pad_token=10, max_length=5, device="cpu"
    )


@pytest.fixture
def env_min_length3():
    return SequenceBase(
        tokens=(-2, -1, 0, 1, 2), pad_token=10, max_length=5, min_length=3, device="cpu"
    )


@pytest.fixture
def env_default():
    return SequenceBase(device="cpu")


@pytest.mark.parametrize(
    "tokens",
    [
        [0, 1],
        [-2, -1, 0, 1, 2],
        (0, 1),
        (-2, -1, 0, 1, 2),
    ],
)
def test__environment_initializes_properly(tokens):
    SequenceBase(tokens=tokens, pad_token=100, device="device")
    assert True


@pytest.mark.parametrize(
    "action_space",
    [
        [(1,), (2,), (3,), (4,), (5,), (-1,)],
    ],
)
def test__get_action_space__returns_expected(env, action_space):
    assert set(action_space) == set(env.action_space)


@pytest.mark.parametrize(
    "env, state, parents_expected, parents_a_expected",
    [
        (
            "env_no_min_length",
            [0, 0, 0, 0, 0],
            [],
            [],
        ),
        (
            "env_no_min_length",
            [1, 0, 0, 0, 0],
            [[0, 0, 0, 0, 0]],
            [(1,)],
        ),
        (
            "env_no_min_length",
            [1, 3, 2, 4, 0],
            [[1, 3, 2, 0, 0]],
            [(4,)],
        ),
        (
            "env_min_length3",
            [0, 0, 0, 0, 0],
            [],
            [],
        ),
        (
            "env_min_length3",
            [1, 0, 0, 0, 0],
            [[0, 0, 0, 0, 0]],
            [(1,)],
        ),
        (
            "env_min_length3",
            [1, 3, 2, 4, 0],
            [[1, 3, 2, 0, 0]],
            [(4,)],
        ),
    ],
)
def test__get_parents__returns_expected(
    env, state, parents_expected, parents_a_expected, request
):
    env = request.getfixturevalue(env)
    state = tlong(state, device=env.device)
    parents_expected = [tlong(parent, device=env.device) for parent in parents_expected]
    parents, parents_a = env.get_parents(state)
    for p, p_e in zip(parents, parents_expected):
        assert torch.equal(p, p_e)
    for p_a, p_a_e in zip(parents_a, parents_a_expected):
        assert p_a == p_a_e


@pytest.mark.parametrize(
    "state, action, next_state",
    [
        (
            [0, 0, 0, 0, 0],
            (1,),
            [1, 0, 0, 0, 0],
        ),
        (
            [1, 0, 0, 0, 0],
            (3,),
            [1, 3, 0, 0, 0],
        ),
        (
            [1, 3, 0, 0, 0],
            (2,),
            [1, 3, 2, 0, 0],
        ),
        (
            [1, 3, 2, 0, 0],
            (1,),
            [1, 3, 2, 1, 0],
        ),
        (
            [1, 3, 2, 1, 0],
            (4,),
            [1, 3, 2, 1, 4],
        ),
        (
            [1, 3, 2, 1, 4],
            (-1,),
            [1, 3, 2, 1, 4],
        ),
        (
            [1, 3, 2, 0, 0],
            (-1,),
            [1, 3, 2, 0, 0],
        ),
    ],
)
def test__step__returns_expected(env, state, action, next_state):
    env.set_state(tlong(state, device=env.device))
    env.step(action)
    assert torch.equal(env.state, tlong(next_state, device=env.device))


@pytest.mark.parametrize(
    "env, state, mask",
    [
        (
            "env_no_min_length",
            [0, 0, 0, 0, 0],
            [False, False, False, False, False, True],
        ),
        ("env_min_length3", [0, 0, 0, 0, 0], [False, False, False, False, False, True]),
        (
            "env_no_min_length",
            [2, 0, 0, 0, 0],
            [False, False, False, False, False, False],
        ),
        ("env_min_length3", [2, 0, 0, 0, 0], [False, False, False, False, False, True]),
        ("env_min_length3", [2, 1, 0, 0, 0], [False, False, False, False, False, True]),
        (
            "env_min_length3",
            [2, 1, 3, 0, 0],
            [False, False, False, False, False, False],
        ),
        ("env_no_min_length", [2, 1, 3, 5, 4], [True, True, True, True, True, False]),
        ("env_min_length3", [2, 1, 3, 5, 4], [True, True, True, True, True, False]),
    ],
)
def test__get_mask_forward__returns_expected(env, state, mask, request):
    env = request.getfixturevalue(env)
    env.set_state(tlong(state, device=env.device))
    assert env.get_mask_invalid_actions_forward() == mask


@pytest.mark.parametrize(
    "states, states_policy",
    [
        (
            [[0, 0, 0, 0, 0], [1, 3, 2, 1, 4]],
            [
                # fmt: off
                [
                    1, 0, 0, 0, 0, 0,
                    1, 0, 0, 0, 0, 0,
                    1, 0, 0, 0, 0, 0,
                    1, 0, 0, 0, 0, 0,
                    1, 0, 0, 0, 0, 0,
                ],
                [
                    0, 1, 0, 0, 0, 0,
                    0, 0, 0, 1, 0, 0,
                    0, 0, 1, 0, 0, 0,
                    0, 1, 0, 0, 0, 0,
                    0, 0, 0, 0, 1, 0,
                ],
                # fmt: on
            ],
        ),
        (
            [[1, 3, 2, 1, 4]],
            [
                # fmt: off
                [
                    0, 1, 0, 0, 0, 0,
                    0, 0, 0, 1, 0, 0,
                    0, 0, 1, 0, 0, 0,
                    0, 1, 0, 0, 0, 0,
                    0, 0, 0, 0, 1, 0,
                ],
                # fmt: on
            ],
        ),
    ],
)
def test__states2policy__returns_expected(env, states, states_policy):
    states = tlong(states, device=env.device)
    states_policy = tlong(states_policy, device=env.device)
    assert torch.equal(states_policy, env.states2policy(states))


@pytest.mark.parametrize(
    "states, states_proxy",
    [
        (
            [[0, 0, 0, 0, 0], [1, 3, 2, 1, 4]],
            [[10, 10, 10, 10, 10], [-2, 0, -1, -2, 1]],
        ),
        (
            [[1, 3, 2, 1, 4]],
            [[-2, 0, -1, -2, 1]],
        ),
    ],
)
def test__states2proxy__returns_expected(env, states, states_proxy):
    states = tlong(states, device=env.device)
    assert env.states2proxy(states) == states_proxy


@pytest.mark.parametrize(
    "state, readable",
    [
        ([0, 0, 0, 0, 0], ""),
        ([1, 3, 2, 0, 0], "-2 0 -1"),
        ([1, 3, 2, 1, 4], "-2 0 -1 -2 1"),
    ],
)
def test__state2readable__returns_expected(env, state, readable):
    state = tlong(state, device=env.device)
    assert env.state2readable(state) == readable


@pytest.mark.parametrize(
    "state, readable",
    [
        ([0, 0, 0, 0, 0], ""),
        ([1, 3, 2, 0, 0], "-2 0 -1"),
        ([1, 3, 2, 1, 4], "-2 0 -1 -2 1"),
    ],
)
def test__readable2state__returns_expected(env, state, readable):
    state = tlong(state, device=env.device)
    assert torch.equal(env.readable2state(readable), state)


class TestSequenceBaseDefault(common.BaseTestsDiscrete):
    """Common tests the default SequenceBase"""

    @pytest.fixture(autouse=True)
    def setup(self, env_default):
        self.env = env_default
        self.repeats = {
            "test__reset__state_is_source": 10,
            "test__forward_actions_have_nonzero_backward_prob": 10,
            "test__backward_actions_have_nonzero_forward_prob": 10,
            "test__trajectories_are_reversible": 10,
            "test__step_random__does_not_sample_invalid_actions_forward": 10,
            "test__step_random__does_not_sample_invalid_actions_backward": 10,
            "test__sample_actions__get_logprobs__return_valid_actions_and_logprobs": 10,
            "test__get_parents_step_get_mask__are_compatible": 10,
            "test__state2readable__is_reversible": 10,
        }
        self.n_states = {
            "test__get_logprobs__all_finite_in_random_forward_transitions": 10,
            "test__get_logprobs__all_finite_in_random_backward_transitions": 10,
        }  # TODO: Populate.
        self.batch_size = {
            "test__sample_actions__get_logprobs__batched_forward_trajectories": 10,
            "test__sample_actions__get_logprobs__batched_backward_trajectories": 10,
            "test__get_logprobs__all_finite_in_accumulated_forward_trajectories": 10,
        }


class TestSequenceBaseNoMinLength(common.BaseTestsDiscrete):
    """Common tests the default SequenceBase"""

    @pytest.fixture(autouse=True)
    def setup(self, env_no_min_length):
        self.env = env_no_min_length
        self.repeats = {
            "test__reset__state_is_source": 10,
            "test__forward_actions_have_nonzero_backward_prob": 10,
            "test__backward_actions_have_nonzero_forward_prob": 10,
            "test__trajectories_are_reversible": 10,
            "test__step_random__does_not_sample_invalid_actions_forward": 10,
            "test__step_random__does_not_sample_invalid_actions_backward": 10,
            "test__sample_actions__get_logprobs__return_valid_actions_and_logprobs": 10,
            "test__get_parents_step_get_mask__are_compatible": 10,
            "test__state2readable__is_reversible": 10,
        }
        self.n_states = {
            "test__get_logprobs__all_finite_in_random_forward_transitions": 10,
            "test__get_logprobs__all_finite_in_random_backward_transitions": 10,
        }  # TODO: Populate.
        self.batch_size = {
            "test__sample_actions__get_logprobs__batched_forward_trajectories": 10,
            "test__sample_actions__get_logprobs__batched_backward_trajectories": 10,
            "test__get_logprobs__all_finite_in_accumulated_forward_trajectories": 10,
        }


class TestSequenceBaseMinLength3(common.BaseTestsDiscrete):
    """Common tests the default SequenceBase"""

    @pytest.fixture(autouse=True)
    def setup(self, env_min_length3):
        self.env = env_min_length3
        self.repeats = {
            "test__reset__state_is_source": 10,
            "test__forward_actions_have_nonzero_backward_prob": 10,
            "test__backward_actions_have_nonzero_forward_prob": 10,
            "test__trajectories_are_reversible": 10,
            "test__step_random__does_not_sample_invalid_actions_forward": 10,
            "test__step_random__does_not_sample_invalid_actions_backward": 10,
            "test__sample_actions__get_logprobs__return_valid_actions_and_logprobs": 10,
            "test__get_parents_step_get_mask__are_compatible": 10,
            "test__state2readable__is_reversible": 10,
        }
        self.n_states = {
            "test__get_logprobs__all_finite_in_random_forward_transitions": 10,
            "test__get_logprobs__all_finite_in_random_backward_transitions": 10,
        }  # TODO: Populate.
        self.batch_size = {
            "test__sample_actions__get_logprobs__batched_forward_trajectories": 10,
            "test__sample_actions__get_logprobs__batched_backward_trajectories": 10,
            "test__get_logprobs__all_finite_in_accumulated_forward_trajectories": 10,
        }
