import common
import pytest
import torch

from gflownet.envs.options import Options
from gflownet.utils.common import tfloat


@pytest.fixture
def env():
    return Options()


@pytest.fixture
def env_with_options():
    return Options(options=["A", "B", "C", "D", "E"])


@pytest.fixture
def env_with_10_options():
    return Options(n_options=10)


@pytest.mark.parametrize(
    "env",
    [
        "env",
        "env_with_options",
        "env_with_10_options",
    ],
)
def test__environment__initializes_properly(env, request):
    env = request.getfixturevalue(env)
    assert True


@pytest.mark.parametrize(
    "action_space",
    [
        [
            (1,),
            (2,),
            (3,),
            (4,),
            (5,),
            (-1,),
        ],
    ],
)
def test__get_action_space__returns_expected(env_with_options, action_space):
    env = env_with_options
    assert set(action_space) == set(env.action_space)


@pytest.mark.parametrize(
    "state, action, next_state",
    [
        # Source state
        (
            [0],
            (1,),
            [1],
        ),
        (
            [0],
            (2,),
            [2],
        ),
        (
            [0],
            (3,),
            [3],
        ),
        # Option selected
        (
            [1],
            (-1,),
            [1],
        ),
        (
            [2],
            (-1,),
            [2],
        ),
        (
            [3],
            (-1,),
            [3],
        ),
    ],
)
def test__step__returns_expected(env, state, action, next_state):
    env.set_state(state)
    env.step(action, skip_mask_check=True)
    assert env.equal(env.state, next_state)


@pytest.mark.parametrize(
    "state, mask_expected",
    [
        # Source state
        (
            [0],
            [False, False, False, True],
        ),
        # Option selected
        (
            [1],
            [True, True, True, False],
        ),
        (
            [2],
            [True, True, True, False],
        ),
        (
            [3],
            [True, True, True, False],
        ),
    ],
)
def test__get_mask_invalid_actions_forward__returns_expected(env, state, mask_expected):
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
            [0],
            [],
            [],
        ),
        # Options selected
        (
            [1],
            [[0]],
            [(1,)],
        ),
        (
            [2],
            [[0]],
            [(2,)],
        ),
        (
            [3],
            [[0]],
            [(3,)],
        ),
    ],
)
def test__get_parents__returns_expected(
    env, state, parents_expected, a_parents_expected
):
    parents, a_parents = env.get_parents(state, done=False)
    for parent, action in zip(parents, a_parents):
        assert action in a_parents_expected
        idx = a_parents_expected.index(action)
        parent_expected = parents_expected[idx]
        assert env.equal(parent, parent_expected)


@pytest.mark.parametrize(
    "batch, exp_tensor",
    [
        (
            [[0], [1], [2], [3]],
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
        ),
    ],
)
def test__states2policy__returns_expected_tensor(env, batch, exp_tensor):
    assert torch.equal(
        env.states2policy(batch),
        tfloat(exp_tensor, device=env.device, float_type=env.float),
    )


class TestOptionsDefault(common.BaseTestsDiscrete):
    """Common tests for default Options environment"""

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
            "test__get_logprobs__all_finite_in_random_forward_transitions": 3,
            "test__get_logprobs__all_finite_in_random_backward_transitions": 3,
        }
        self.batch_size = {
            "test__sample_actions__get_logprobs__batched_forward_trajectories": 10,
            "test__sample_actions__get_logprobs__batched_backward_trajectories": 10,
            "test__get_logprobs__all_finite_in_accumulated_forward_trajectories": 10,
        }
