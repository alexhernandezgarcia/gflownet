import common
import pytest
import torch

from gflownet.envs.choices import Choices
from gflownet.utils.common import tfloat


@pytest.fixture
def env_default():
    return Choices()


@pytest.fixture
def env_with_replacement():
    return Choices(
        options=["A", "B", "C", "D", "E"], max_selection=3, with_replacement=True
    )


@pytest.fixture
def env_without_replacement():
    return Choices(
        options=["A", "B", "C", "D", "E"], max_selection=3, with_replacement=False
    )


@pytest.mark.parametrize(
    "env",
    [
        "env_default",
        "env_with_replacement",
        "env_without_replacement",
    ],
)
def test__environment__initializes_properly(env, request):
    env = request.getfixturevalue(env)
    assert True


@pytest.mark.parametrize(
    "env, has_constraints_exp",
    [
        ("env_default", False),
        ("env_with_replacement", False),
        ("env_without_replacement", True),
    ],
)
def test__constraints_are_as_expected(env, has_constraints_exp, request):
    env = request.getfixturevalue(env)
    assert env.has_constraints == has_constraints_exp


@pytest.mark.repeat(10)
@pytest.mark.parametrize(
    "env",
    [
        "env_default",
        "env_with_replacement",
        "env_without_replacement",
    ],
)
def test__trajectory_random__does_not_crash_and_reaches_done(env, request):
    env = request.getfixturevalue(env)
    env.reset()
    env.trajectory_random()
    assert env.done


@pytest.mark.repeat(100)
@pytest.mark.parametrize(
    "env",
    [
        "env_without_replacement",
    ],
)
def test__envs_without_replacement_do_not_repeat_elements(env, request):
    env = request.getfixturevalue(env)
    env.reset()
    env.trajectory_random()
    assert len(env.get_options()) == env.max_selection


@pytest.mark.repeat(10)
@pytest.mark.parametrize(
    "env",
    [
        "env_default",
        "env_with_replacement",
        "env_without_replacement",
    ],
)
def test__trajectory_backwards_random__does_not_crash_and_reaches_source(env, request):
    env = request.getfixturevalue(env)
    env.reset()
    env.trajectory_random()
    assert env.done
    env.trajectory_random(backward=True)
    assert env.is_source()


@pytest.mark.parametrize(
    "env, states, states_policy_exp",
    [
        (
            "env_default",
            [
                {
                    "_active": -1,
                    "_toggle": 0,
                    "_dones": [0, 0],
                    "_envs_unique": [0, 0],
                    0: [0],
                    1: [0],
                },
                {
                    "_active": 0,
                    "_toggle": 0,
                    "_dones": [0, 0],
                    "_envs_unique": [0, 0],
                    0: [0],
                    1: [0],
                },
                {
                    "_active": 0,
                    "_toggle": 0,
                    "_dones": [0, 0],
                    "_envs_unique": [0, 0],
                    0: [2],
                    1: [0],
                },
                {
                    "_active": 0,
                    "_toggle": 0,
                    "_dones": [1, 0],
                    "_envs_unique": [0, 0],
                    0: [2],
                    1: [0],
                },
                {
                    "_active": -1,
                    "_toggle": 0,
                    "_dones": [1, 0],
                    "_envs_unique": [0, 0],
                    0: [2],
                    1: [0],
                },
                {
                    "_active": 1,
                    "_toggle": 0,
                    "_dones": [1, 0],
                    "_envs_unique": [0, 0],
                    0: [2],
                    1: [0],
                },
                {
                    "_active": 1,
                    "_toggle": 0,
                    "_dones": [1, 0],
                    "_envs_unique": [0, 0],
                    0: [2],
                    1: [1],
                },
                {
                    "_active": 1,
                    "_toggle": 0,
                    "_dones": [1, 1],
                    "_envs_unique": [0, 0],
                    0: [2],
                    1: [1],
                },
                {
                    "_active": -1,
                    "_toggle": 0,
                    "_dones": [1, 1],
                    "_envs_unique": [0, 0],
                    0: [2],
                    1: [1],
                },
            ],
            torch.stack(
                [
                    torch.tensor(
                        [
                            # fmt: off
                            -1.0, # FLAG
                            # OPTIONS
                            0.0, 0.0, 0.0,
                            # fmt: on
                        ],
                        dtype=torch.float,
                    ),
                    torch.tensor(
                        [
                            # fmt: off
                            0.0, # FLAG
                            # OPTIONS
                            0.0, 0.0, 0.0,
                            # fmt: on
                        ],
                        dtype=torch.float,
                    ),
                    torch.tensor(
                        [
                            # fmt: off
                            0.0, # FLAG
                            # OPTIONS
                            0.0, 1.0, 0.0,
                            # fmt: on
                        ],
                        dtype=torch.float,
                    ),
                    torch.tensor(
                        [
                            # fmt: off
                            1.0, # FLAG
                            # OPTIONS
                            0.0, 1.0, 0.0,
                            # fmt: on
                        ],
                        dtype=torch.float,
                    ),
                    torch.tensor(
                        [
                            # fmt: off
                            -1.0, # FLAG
                            # OPTIONS
                            0.0, 1.0, 0.0,
                            # fmt: on
                        ],
                        dtype=torch.float,
                    ),
                    torch.tensor(
                        [
                            # fmt: off
                            0.0, # FLAG
                            # OPTIONS
                            0.0, 1.0, 0.0,
                            # fmt: on
                        ],
                        dtype=torch.float,
                    ),
                    torch.tensor(
                        [
                            # fmt: off
                            0.0, # FLAG
                            # OPTIONS
                            1.0, 1.0, 0.0,
                            # fmt: on
                        ],
                        dtype=torch.float,
                    ),
                    torch.tensor(
                        [
                            # fmt: off
                            1.0, # FLAG
                            # OPTIONS
                            1.0, 1.0, 0.0,
                            # fmt: on
                        ],
                        dtype=torch.float,
                    ),
                    torch.tensor(
                        [
                            # fmt: off
                            -1.0, # FLAG
                            # OPTIONS
                            1.0, 1.0, 0.0,
                            # fmt: on
                        ],
                        dtype=torch.float,
                    ),
                ],
                dim=0,
            ),
        ),
        (
            "env_with_replacement",
            [
                {
                    "_active": -1,
                    "_toggle": 0,
                    "_dones": [0, 0, 0],
                    "_envs_unique": [0, 0, 0],
                    0: [0],
                    1: [0],
                    2: [0],
                },
                {
                    "_active": 0,
                    "_toggle": 0,
                    "_dones": [0, 0, 0],
                    "_envs_unique": [0, 0, 0],
                    0: [0],
                    1: [0],
                    2: [0],
                },
                {
                    "_active": 0,
                    "_toggle": 0,
                    "_dones": [0, 0, 0],
                    "_envs_unique": [0, 0, 0],
                    0: [2],
                    1: [0],
                    2: [0],
                },
                {
                    "_active": 0,
                    "_toggle": 0,
                    "_dones": [1, 0, 0],
                    "_envs_unique": [0, 0, 0],
                    0: [2],
                    1: [0],
                    2: [0],
                },
                {
                    "_active": -1,
                    "_toggle": 0,
                    "_dones": [1, 1, 1],
                    "_envs_unique": [0, 0, 0],
                    0: [2],
                    1: [2],
                    2: [5],
                },
                {
                    "_active": 2,
                    "_toggle": 0,
                    "_dones": [1, 1, 0],
                    "_envs_unique": [0, 0, 0],
                    0: [2],
                    1: [2],
                    2: [0],
                },
            ],
            torch.stack(
                [
                    torch.tensor(
                        [
                            # fmt: off
                            -1.0, # FLAG
                            # OPTIONS
                            0.0, 0.0, 0.0, 0.0, 0.0,
                            # fmt: on
                        ],
                        dtype=torch.float,
                    ),
                    torch.tensor(
                        [
                            # fmt: off
                            0.0, # FLAG
                            # OPTIONS
                            0.0, 0.0, 0.0, 0.0, 0.0,
                            # fmt: on
                        ],
                        dtype=torch.float,
                    ),
                    torch.tensor(
                        [
                            # fmt: off
                            0.0, # FLAG
                            # OPTIONS
                            0.0, 1.0, 0.0, 0.0, 0.0,
                            # fmt: on
                        ],
                        dtype=torch.float,
                    ),
                    torch.tensor(
                        [
                            # fmt: off
                            1.0, # FLAG
                            # OPTIONS
                            0.0, 1.0, 0.0, 0.0, 0.0,
                            # fmt: on
                        ],
                        dtype=torch.float,
                    ),
                    torch.tensor(
                        [
                            # fmt: off
                            -1.0, # FLAG
                            # OPTIONS
                            0.0, 2.0, 0.0, 0.0, 1.0,
                            # fmt: on
                        ],
                        dtype=torch.float,
                    ),
                    torch.tensor(
                        [
                            # fmt: off
                            0.0, # FLAG
                            # OPTIONS
                            0.0, 2.0, 0.0, 0.0, 0.0,
                            # fmt: on
                        ],
                        dtype=torch.float,
                    ),
                ],
                dim=0,
            ),
        ),
    ],
)
def test__states2policy__returns_expected(env, states, states_policy_exp, request):
    env = request.getfixturevalue(env)
    assert torch.equal(states_policy_exp, env.states2policy(states))


class TestChoicesDefault(common.BaseTestsDiscrete):
    """Common tests for default Choices environment"""

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
        }
        self.batch_size = {
            "test__sample_actions__get_logprobs__batched_forward_trajectories": 10,
            "test__sample_actions__get_logprobs__batched_backward_trajectories": 10,
            "test__get_logprobs__all_finite_in_accumulated_forward_trajectories": 10,
        }


class TestChoicesWithReplacement(common.BaseTestsDiscrete):
    """Common tests for Choices with replacement environment"""

    @pytest.fixture(autouse=True)
    def setup(self, env_with_replacement):
        self.env = env_with_replacement
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
        }
        self.batch_size = {
            "test__sample_actions__get_logprobs__batched_forward_trajectories": 10,
            "test__sample_actions__get_logprobs__batched_backward_trajectories": 10,
            "test__get_logprobs__all_finite_in_accumulated_forward_trajectories": 10,
        }


class TestChoicesWithoutReplacement(common.BaseTestsDiscrete):
    """Common tests for Choices with replacement environment"""

    @pytest.fixture(autouse=True)
    def setup(self, env_without_replacement):
        self.env = env_without_replacement
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
        }
        self.batch_size = {
            "test__sample_actions__get_logprobs__batched_forward_trajectories": 10,
            "test__sample_actions__get_logprobs__batched_backward_trajectories": 10,
            "test__get_logprobs__all_finite_in_accumulated_forward_trajectories": 10,
        }
