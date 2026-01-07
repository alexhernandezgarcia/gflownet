import common
import pytest
import torch

from gflownet.envs.grid import Grid
from gflownet.utils.common import tfloat


@pytest.fixture
def env():
    return Grid(n_dim=3, length=5, cell_min=-1.0, cell_max=1.0)


@pytest.fixture
def env_default():
    return Grid()


@pytest.fixture
def env_extended_action_space_2d():
    return Grid(
        n_dim=2,
        length=5,
        max_increment=2,
        max_dim_per_action=-1,
        cell_min=-1.0,
        cell_max=1.0,
    )


@pytest.fixture
def env_extended_action_space_3d():
    return Grid(
        n_dim=3,
        length=5,
        max_increment=2,
        max_dim_per_action=2,
        cell_min=-1.0,
        cell_max=1.0,
    )


@pytest.fixture
def config_path():
    return "../../../config/env/grid.yaml"


@pytest.mark.parametrize(
    "state, state2proxy",
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
def test__state2proxy__returns_expected(env, state, state2proxy):
    assert torch.equal(
        tfloat(state2proxy, device=env.device, float_type=env.float),
        env.state2proxy(state)[0],
    )


@pytest.mark.parametrize(
    "states, states2proxy",
    [
        (
            [[0, 0, 0], [4, 4, 4], [1, 2, 3], [4, 0, 1]],
            [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], [-0.5, 0.0, 0.5], [1.0, -1.0, -0.5]],
        ),
    ],
)
def test__states2proxy__returns_expected(env, states, states2proxy):
    assert torch.equal(torch.Tensor(states2proxy), env.states2proxy(states))


@pytest.mark.parametrize(
    "env, action_space",
    [
        (
            "env_extended_action_space_2d",
            [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2)],
        ),
        (
            "env_extended_action_space_3d",
            [
                (0, 0, 0),
                (1, 0, 0),
                (0, 1, 0),
                (0, 0, 1),
                (1, 1, 0),
                (1, 0, 1),
                (0, 1, 1),
                (2, 0, 0),
                (0, 2, 0),
                (0, 0, 2),
                (2, 1, 0),
                (2, 0, 1),
                (1, 2, 0),
                (1, 0, 2),
                (0, 2, 1),
                (0, 1, 2),
                (2, 2, 0),
                (2, 0, 2),
                (0, 2, 2),
            ],
        ),
    ],
)
def test__get_action_space__returns_expected(env, action_space, request):
    env = request.getfixturevalue(env)
    assert set(action_space) == set(env.action_space)


@pytest.mark.parametrize(
    "env, state, actions_valid_exp",
    [
        (
            "env_extended_action_space_2d",
            [0, 0],
            {(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (2, 1), (1, 2), (2, 2)},
        ),
        (
            "env_extended_action_space_2d",
            [2, 2],
            {(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (2, 1), (1, 2), (2, 2)},
        ),
        (
            "env_extended_action_space_2d",
            [2, 3],
            {(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (2, 1)},
        ),
        (
            "env_extended_action_space_2d",
            [4, 1],
            {(0, 0), (0, 1), (0, 2)},
        ),
        (
            "env_extended_action_space_2d",
            [4, 3],
            {(0, 0), (0, 1)},
        ),
        (
            "env_extended_action_space_3d",
            [0, 0, 0],
            {
                (0, 0, 0),
                (1, 0, 0),
                (0, 1, 0),
                (0, 0, 1),
                (1, 1, 0),
                (1, 0, 1),
                (0, 1, 1),
                (2, 0, 0),
                (0, 2, 0),
                (0, 0, 2),
                (2, 1, 0),
                (2, 0, 1),
                (1, 2, 0),
                (1, 0, 2),
                (0, 2, 1),
                (0, 1, 2),
                (2, 2, 0),
                (2, 0, 2),
                (0, 2, 2),
            },
        ),
        (
            "env_extended_action_space_3d",
            [4, 3, 2],
            {
                (0, 0, 0),
                (0, 1, 0),
                (0, 0, 1),
                (0, 1, 1),
                (0, 0, 2),
                (0, 1, 2),
            },
        ),
    ],
)
def test__get_mask_invalid_actions_forward__masks_expected_actions(
    env, state, actions_valid_exp, request
):
    env = request.getfixturevalue(env)
    assert set(actions_valid_exp) == set(env.get_valid_actions(state=state))
    env.set_state(state)
    assert set(actions_valid_exp) == set(env.get_valid_actions())
    env.done = True
    assert set(env.get_valid_actions()) == set()


@pytest.mark.parametrize(
    "env, state, actions_valid_exp",
    [
        (
            "env_extended_action_space_2d",
            [0, 0],
            set(),
        ),
        (
            "env_extended_action_space_2d",
            [2, 2],
            {(1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2)},
        ),
        (
            "env_extended_action_space_2d",
            [1, 3],
            {(1, 0), (0, 1), (1, 1), (0, 2), (1, 2)},
        ),
        (
            "env_extended_action_space_2d",
            [0, 1],
            {(0, 1)},
        ),
        (
            "env_extended_action_space_3d",
            [0, 0, 0],
            set(),
        ),
        (
            "env_extended_action_space_3d",
            [4, 3, 2],
            {
                (1, 0, 0),
                (0, 1, 0),
                (0, 0, 1),
                (1, 1, 0),
                (1, 0, 1),
                (0, 1, 1),
                (2, 0, 0),
                (0, 2, 0),
                (0, 0, 2),
                (2, 1, 0),
                (2, 0, 1),
                (1, 2, 0),
                (1, 0, 2),
                (0, 2, 1),
                (0, 1, 2),
                (2, 2, 0),
                (2, 0, 2),
                (0, 2, 2),
            },
        ),
        (
            "env_extended_action_space_3d",
            [1, 0, 2],
            {
                (1, 0, 0),
                (0, 0, 1),
                (1, 0, 1),
                (0, 0, 2),
                (1, 0, 2),
            },
        ),
    ],
)
def test__get_mask_invalid_actions_backward__masks_expected_actions(
    env, state, actions_valid_exp, request
):
    env = request.getfixturevalue(env)
    assert set(actions_valid_exp) == set(
        env.get_valid_actions(state=state, backward=True)
    )
    env.set_state(state)
    assert set(actions_valid_exp) == set(env.get_valid_actions(backward=True))


@pytest.mark.parametrize(
    "state, parents_expected, parents_a_expected",
    [
        (
            [0, 0],
            [],
            [],
        ),
        (
            [1, 0],
            [[0, 0]],
            [(1, 0)],
        ),
        (
            [1, 1],
            [[0, 1], [1, 0], [0, 0]],
            [(1, 0), (0, 1), (1, 1)],
        ),
        (
            [3, 4],
            [[2, 4], [1, 4], [3, 3], [2, 3], [1, 3], [3, 2], [2, 3], [1, 2]],
            [(1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2)],
        ),
    ],
)
def test__get_parents__returns_expected(
    env_extended_action_space_2d, state, parents_expected, parents_a_expected
):
    env = env_extended_action_space_2d
    parents, parents_a = env.get_parents(state)
    assert len(parents) == len(parents_expected)
    assert len(parents_a) == len(parents_a_expected)
    # Create dictionaries of parent_action: parent for comparison
    parents_actions_exp_dict = {}
    for parent, action in zip(parents_expected, parents_a_expected):
        parents_actions_exp_dict[action] = tuple(parent.copy())
    parents_actions_dict = {}
    for parent, action in zip(parents, parents_a):
        parents_actions_dict[action] = tuple(parent.copy())

    assert sorted(parents_actions_exp_dict) == sorted(parents_actions_dict)


class TestGridBasic(common.BaseTestsDiscrete):
    """Common tests for 5x5 Grid with standard action space."""

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


class TestGridDefaults(common.BaseTestsDiscrete):
    """Common tests for 5x5 Grid with standard action space."""

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
            "test__get_logprobs__all_finite_in_random_forward_transitions": 5,
            "test__get_logprobs__all_finite_in_random_backward_transitions": 5,
        }  # TODO: Populate.
        self.batch_size = {
            "test__sample_actions__get_logprobs__batched_forward_trajectories": 10,
            "test__sample_actions__get_logprobs__batched_backward_trajectories": 10,
            "test__get_logprobs__all_finite_in_accumulated_forward_trajectories": 10,
        }


class TestGridExtended2D(common.BaseTestsDiscrete):
    """Common tests for 5x5 Grid with extended action space."""

    @pytest.fixture(autouse=True)
    def setup(self, env_extended_action_space_2d):
        self.env = env_extended_action_space_2d
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


class TestGridExtended3D(common.BaseTestsDiscrete):
    """Common tests for 5x5 Grid with extended action space."""

    @pytest.fixture(autouse=True)
    def setup(self, env_extended_action_space_3d):
        self.env = env_extended_action_space_3d
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
