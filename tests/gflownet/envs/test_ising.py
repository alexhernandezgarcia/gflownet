import common
import numpy as np
import pytest

from gflownet.envs.ising import SET_SPIN, TOGGLE_VARIABLE, TOGGLED, UNSET, Ising


@pytest.fixture
def env():
    return Ising()


def test__environment_initializes_properly():
    env = Ising()
    assert True


@pytest.mark.parametrize(
    "action_space",
    [
        [
            (TOGGLE_VARIABLE, 0),
            (TOGGLE_VARIABLE, 1),
            (TOGGLE_VARIABLE, 2),
            (TOGGLE_VARIABLE, 3),
            (TOGGLE_VARIABLE, 4),
            (TOGGLE_VARIABLE, 5),
            (TOGGLE_VARIABLE, 6),
            (TOGGLE_VARIABLE, 7),
            (TOGGLE_VARIABLE, 8),
            (TOGGLE_VARIABLE, 9),
            (TOGGLE_VARIABLE, 10),
            (TOGGLE_VARIABLE, 11),
            (TOGGLE_VARIABLE, 12),
            (TOGGLE_VARIABLE, 13),
            (TOGGLE_VARIABLE, 14),
            (TOGGLE_VARIABLE, 15),
            (SET_SPIN, -1),
            (SET_SPIN, 1),
            (-1, -1),
        ],
    ],
)
def test__get_action_space__returns_expected(env, action_space):
    assert set(action_space) == set(env.action_space)


@pytest.mark.parametrize(
    "state, action, next_state",
    [
        # Toggle variable from source
        (
            [
                [UNSET, UNSET, UNSET, UNSET],
                [UNSET, UNSET, UNSET, UNSET],
                [UNSET, UNSET, UNSET, UNSET],
                [UNSET, UNSET, UNSET, UNSET],
            ],
            (TOGGLE_VARIABLE, 0),
            [
                [TOGGLED, UNSET, UNSET, UNSET],
                [UNSET, UNSET, UNSET, UNSET],
                [UNSET, UNSET, UNSET, UNSET],
                [UNSET, UNSET, UNSET, UNSET],
            ],
        ),
        # Set spin from toggled
        (
            [
                [TOGGLED, UNSET, UNSET, UNSET],
                [UNSET, UNSET, UNSET, UNSET],
                [UNSET, UNSET, UNSET, UNSET],
                [UNSET, UNSET, UNSET, UNSET],
            ],
            (SET_SPIN, -1),
            [
                [-2, UNSET, UNSET, UNSET],
                [UNSET, UNSET, UNSET, UNSET],
                [UNSET, UNSET, UNSET, UNSET],
                [UNSET, UNSET, UNSET, UNSET],
            ],
        ),
        # Set spin from toggled
        (
            [
                [TOGGLED, UNSET, UNSET, UNSET],
                [UNSET, UNSET, UNSET, UNSET],
                [UNSET, UNSET, UNSET, UNSET],
                [UNSET, UNSET, UNSET, UNSET],
            ],
            (SET_SPIN, 1),
            [
                [2, UNSET, UNSET, UNSET],
                [UNSET, UNSET, UNSET, UNSET],
                [UNSET, UNSET, UNSET, UNSET],
                [UNSET, UNSET, UNSET, UNSET],
            ],
        ),
        # Toggle after setting sin
        (
            [
                [-2, UNSET, UNSET, UNSET],
                [UNSET, UNSET, UNSET, UNSET],
                [UNSET, UNSET, UNSET, UNSET],
                [UNSET, UNSET, UNSET, UNSET],
            ],
            (TOGGLE_VARIABLE, 0),
            [
                [-1, UNSET, UNSET, UNSET],
                [UNSET, UNSET, UNSET, UNSET],
                [UNSET, UNSET, UNSET, UNSET],
                [UNSET, UNSET, UNSET, UNSET],
            ],
        ),
        # Toggle after setting sin
        (
            [
                [2, UNSET, UNSET, UNSET],
                [UNSET, UNSET, UNSET, UNSET],
                [UNSET, UNSET, UNSET, UNSET],
                [UNSET, UNSET, UNSET, UNSET],
            ],
            (TOGGLE_VARIABLE, 0),
            [
                [1, UNSET, UNSET, UNSET],
                [UNSET, UNSET, UNSET, UNSET],
                [UNSET, UNSET, UNSET, UNSET],
                [UNSET, UNSET, UNSET, UNSET],
            ],
        ),
        # Toggle variable from generic state
        (
            [
                [-1, UNSET, -1, UNSET],
                [UNSET, -1, UNSET, UNSET],
                [UNSET, 1, UNSET, UNSET],
                [-1, UNSET, UNSET, UNSET],
            ],
            (TOGGLE_VARIABLE, 6),
            [
                [-1, UNSET, -1, UNSET],
                [UNSET, -1, TOGGLED, UNSET],
                [UNSET, 1, UNSET, UNSET],
                [-1, UNSET, UNSET, UNSET],
            ],
        ),
        # Set spin after toggling generic state
        (
            [
                [-1, UNSET, -1, UNSET],
                [UNSET, -1, TOGGLED, UNSET],
                [UNSET, 1, UNSET, UNSET],
                [-1, UNSET, UNSET, UNSET],
            ],
            (SET_SPIN, 1),
            [
                [-1, UNSET, -1, UNSET],
                [UNSET, -1, 2, UNSET],
                [UNSET, 1, UNSET, UNSET],
                [-1, UNSET, UNSET, UNSET],
            ],
        ),
    ],
)
def test__step__returns_expected(env, state, action, next_state):
    state = np.array(state)
    next_state = np.array(next_state)
    env.set_state(state)
    env.step(action, skip_mask_check=True)
    assert env.equal(env.state, next_state)


@pytest.mark.parametrize(
    "state, mask_expected",
    [
        # Source state
        (
            [
                [UNSET, UNSET, UNSET, UNSET],
                [UNSET, UNSET, UNSET, UNSET],
                [UNSET, UNSET, UNSET, UNSET],
                [UNSET, UNSET, UNSET, UNSET],
            ],
            # fmt: off
            [
                False, False, False, False,
                False, False, False, False,
                False, False, False, False,
                False, False, False, False,
                True, True, 
                True,
            ]
            # fmt: on
        ),
        # Variable toggled but unset
        (
            [
                [TOGGLED, UNSET, UNSET, UNSET],
                [UNSET, UNSET, UNSET, UNSET],
                [UNSET, UNSET, UNSET, UNSET],
                [UNSET, UNSET, UNSET, UNSET],
            ],
            # fmt: off
            [
                True, True, True, True,
                True, True, True, True,
                True, True, True, True,
                True, True, True, True,
                False, False, 
                True,
            ]
            # fmt: on
        ),
        # Variable set and toggled
        (
            [
                [-2, UNSET, UNSET, UNSET],
                [UNSET, UNSET, UNSET, UNSET],
                [UNSET, UNSET, UNSET, UNSET],
                [UNSET, UNSET, UNSET, UNSET],
            ],
            # fmt: off
            [
                False, True, True, True,
                True, True, True, True,
                True, True, True, True,
                True, True, True, True,
                True, True, 
                True,
            ]
            # fmt: on
        ),
        # Generic neutral state
        (
            [
                [-1, UNSET, -1, UNSET],
                [UNSET, -1, UNSET, UNSET],
                [UNSET, 1, UNSET, UNSET],
                [-1, UNSET, UNSET, UNSET],
            ],
            # fmt: off
            [
                True, False, True, False,
                False, True, False, False,
                False, True, False, False,
                True, False, False, False,
                True, True, 
                True,
            ]
            # fmt: on
        ),
        # Generic state with toggled but unset variable
        (
            [
                [-1, UNSET, -1, UNSET],
                [UNSET, -1, TOGGLED, UNSET],
                [UNSET, 1, UNSET, UNSET],
                [-1, UNSET, UNSET, UNSET],
            ],
            # fmt: off
            [
                True, True, True, True,
                True, True, True, True,
                True, True, True, True,
                True, True, True, True,
                False, False, 
                True,
            ]
            # fmt: on
        ),
        # Generic state with set and toggled variable
        (
            [
                [-1, UNSET, -1, UNSET],
                [UNSET, -1, 2, UNSET],
                [UNSET, 1, UNSET, UNSET],
                [-1, UNSET, UNSET, UNSET],
            ],
            # fmt: off
            [
                True, True, True, True,
                True, True, False, True,
                True, True, True, True,
                True, True, True, True,
                True, True, 
                True,
            ]
            # fmt: on
        ),
        # Terminating state
        (
            [
                [-1, 1, -1, -1],
                [-1, -1, -1, 1],
                [1, 1, 1, -1],
                [-1, 1, -1, 1],
            ],
            # fmt: off
            [
                True, True, True, True,
                True, True, True, True,
                True, True, True, True,
                True, True, True, True,
                True, True, 
                False,
            ]
            # fmt: on
        ),
    ],
)
def test__get_mask_invalid_actions_forward__returns_expected(env, state, mask_expected):
    state = np.array(state)
    mask = env.get_mask_invalid_actions_forward(state, done=False)
    assert mask == mask_expected
    env.set_state(state)
    mask = env.get_mask_invalid_actions_forward()
    assert mask == mask_expected


@pytest.mark.parametrize(
    "state, parents_expected, a_parents_expected",
    [
        # Source state
        (
            [
                [UNSET, UNSET, UNSET, UNSET],
                [UNSET, UNSET, UNSET, UNSET],
                [UNSET, UNSET, UNSET, UNSET],
                [UNSET, UNSET, UNSET, UNSET],
            ],
            [],
            [],
        ),
        # Variable toggled but unset
        (
            [
                [TOGGLED, UNSET, UNSET, UNSET],
                [UNSET, UNSET, UNSET, UNSET],
                [UNSET, UNSET, UNSET, UNSET],
                [UNSET, UNSET, UNSET, UNSET],
            ],
            [
                [
                    [UNSET, UNSET, UNSET, UNSET],
                    [UNSET, UNSET, UNSET, UNSET],
                    [UNSET, UNSET, UNSET, UNSET],
                    [UNSET, UNSET, UNSET, UNSET],
                ]
            ],
            [(TOGGLE_VARIABLE, 0)],
        ),
        # Variable set and toggled
        (
            [
                [-2, UNSET, UNSET, UNSET],
                [UNSET, UNSET, UNSET, UNSET],
                [UNSET, UNSET, UNSET, UNSET],
                [UNSET, UNSET, UNSET, UNSET],
            ],
            [
                [
                    [TOGGLED, UNSET, UNSET, UNSET],
                    [UNSET, UNSET, UNSET, UNSET],
                    [UNSET, UNSET, UNSET, UNSET],
                    [UNSET, UNSET, UNSET, UNSET],
                ]
            ],
            [(SET_SPIN, -1)],
        ),
        # Generic neutral state
        (
            [
                [-1, UNSET, -1, UNSET],
                [UNSET, -1, UNSET, UNSET],
                [UNSET, 1, UNSET, UNSET],
                [-1, UNSET, UNSET, UNSET],
            ],
            [
                [
                    [-2, UNSET, -1, UNSET],
                    [UNSET, -1, UNSET, UNSET],
                    [UNSET, 1, UNSET, UNSET],
                    [-1, UNSET, UNSET, UNSET],
                ],
                [
                    [-1, UNSET, -2, UNSET],
                    [UNSET, -1, UNSET, UNSET],
                    [UNSET, 1, UNSET, UNSET],
                    [-1, UNSET, UNSET, UNSET],
                ],
                [
                    [-1, UNSET, -1, UNSET],
                    [UNSET, -2, UNSET, UNSET],
                    [UNSET, 1, UNSET, UNSET],
                    [-1, UNSET, UNSET, UNSET],
                ],
                [
                    [-1, UNSET, -1, UNSET],
                    [UNSET, -1, UNSET, UNSET],
                    [UNSET, 2, UNSET, UNSET],
                    [-1, UNSET, UNSET, UNSET],
                ],
                [
                    [-1, UNSET, -1, UNSET],
                    [UNSET, -1, UNSET, UNSET],
                    [UNSET, 1, UNSET, UNSET],
                    [-2, UNSET, UNSET, UNSET],
                ],
            ],
            [
                (TOGGLE_VARIABLE, 0),
                (TOGGLE_VARIABLE, 2),
                (TOGGLE_VARIABLE, 5),
                (TOGGLE_VARIABLE, 9),
                (TOGGLE_VARIABLE, 12),
            ],
        ),
        # Generic state with toggled but unset variable
        (
            [
                [-1, UNSET, -1, UNSET],
                [UNSET, -1, TOGGLED, UNSET],
                [UNSET, 1, UNSET, UNSET],
                [-1, UNSET, UNSET, UNSET],
            ],
            [
                [
                    [-1, UNSET, -1, UNSET],
                    [UNSET, -1, UNSET, UNSET],
                    [UNSET, 1, UNSET, UNSET],
                    [-1, UNSET, UNSET, UNSET],
                ],
            ],
            [(TOGGLE_VARIABLE, 6)],
        ),
        # Generic state with set and toggled variable
        (
            [
                [-1, UNSET, -1, UNSET],
                [UNSET, -1, 2, UNSET],
                [UNSET, 1, UNSET, UNSET],
                [-1, UNSET, UNSET, UNSET],
            ],
            [
                [
                    [-1, UNSET, -1, UNSET],
                    [UNSET, -1, TOGGLED, UNSET],
                    [UNSET, 1, UNSET, UNSET],
                    [-1, UNSET, UNSET, UNSET],
                ],
            ],
            [(SET_SPIN, 1)],
        ),
    ],
)
def test__get_parents__returns_expected(
    env, state, parents_expected, a_parents_expected
):
    state = np.array(state)
    parents, a_parents = env.get_parents(state, done=False)
    for parent, action in zip(parents, a_parents):
        assert action in a_parents_expected
        idx = a_parents_expected.index(action)
        parent = np.array(parent)
        parent_expected = np.array(parents_expected[idx])
        assert env.equal(parent, parent_expected)


class TestIsingBasic(common.BaseTestsDiscrete):
    """Common tests for 4x4 Ising model"""

    @pytest.fixture(autouse=True)
    def setup(self, env):
        self.env = env
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
