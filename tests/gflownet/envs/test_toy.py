import common
import pytest
import torch

from gflownet.envs.toy import Toy


@pytest.fixture
def env():
    return Toy()


def test__environment_initializes_properly():
    env = Toy()
    assert env is not None
    assert True


@pytest.mark.parametrize(
    "action_space",
    [
        [
            # fmt: off
            (0, 1), (0, 2),
            (1, 3),
            (2, 3), (2, 4),
            (3, 5),
            (4, 6),
            (5, 7), (5, 8),
            (6, 8), (6, 10),
            (7, 9),
            (8, 9),
            (-1, -1),
            # fmt: on
        ],
    ],
)
def test__get_action_space__returns_expected(env, action_space):
    assert set(action_space) == set(env.action_space)
    assert env.action_space[-1] == env.eos


@pytest.mark.parametrize(
    "state, parents_expected, parents_a_expected",
    [
        (
            [0],
            [],
            [],
        ),
        (
            [1],
            [[0]],
            [(0, 1)],
        ),
        (
            [2],
            [[0]],
            [(0, 2)],
        ),
        (
            [3],
            [[1], [2]],
            [(1, 3), (2, 3)],
        ),
        (
            [4],
            [[2]],
            [(2, 4)],
        ),
        (
            [5],
            [[3]],
            [(3, 5)],
        ),
        (
            [6],
            [[4]],
            [(4, 6)],
        ),
        (
            [7],
            [[5]],
            [(5, 7)],
        ),
        (
            [8],
            [[5], [6]],
            [(5, 8), (6, 8)],
        ),
        (
            [9],
            [[7], [8]],
            [(7, 9), (8, 9)],
        ),
        (
            [10],
            [[6]],
            [(6, 10)],
        ),
    ],
)
def test__get_parents__returns_expected(
    env, state, parents_expected, parents_a_expected
):
    parents, parents_a = env.get_parents(state)
    for p, p_e in zip(parents, parents_expected):
        assert p == p_e
    for p_a, p_a_e in zip(parents_a, parents_a_expected):
        assert p_a == p_a_e


@pytest.mark.parametrize(
    "state, mask_expected",
    [
        (
            [0],
            # fmt: off
            [
                False, False,
                True,
                True, True,
                True,
                True,
                True, True,
                True, True,
                True,
                True,
                True,
            ],
            # fmt: on
        ),
        (
            [1],
            # fmt: off
            [
                True, True,
                False,
                True, True,
                True,
                True,
                True, True,
                True, True,
                True,
                True,
                True,
            ],
            # fmt: on
        ),
        (
            [2],
            # fmt: off
            [
                True, True,
                True,
                False, False,
                True,
                True,
                True, True,
                True, True,
                True,
                True,
                True,
            ],
            # fmt: on
        ),
        (
            [3],
            # fmt: off
            [
                True, True,
                True,
                True, True,
                False,
                True,
                True, True,
                True, True,
                True,
                True,
                False,
            ],
            # fmt: on
        ),
        (
            [4],
            # fmt: off
            [
                True, True,
                True,
                True, True,
                True,
                False,
                True, True,
                True, True,
                True,
                True,
                False,
            ],
            # fmt: on
        ),
        (
            [5],
            # fmt: off
            [
                True, True,
                True,
                True, True,
                True,
                True,
                False, False,
                True, True,
                True,
                True,
                True,
            ],
            # fmt: on
        ),
        (
            [6],
            # fmt: off
            [
                True, True,
                True,
                True, True,
                True,
                True,
                True, True,
                False, False,
                True,
                True,
                False,
            ],
            # fmt: on
        ),
        (
            [7],
            # fmt: off
            [
                True, True,
                True,
                True, True,
                True,
                True,
                True, True,
                True, True,
                False,
                True,
                True,
            ],
            # fmt: on
        ),
        (
            [8],
            # fmt: off
            [
                True, True,
                True,
                True, True,
                True,
                True,
                True, True,
                True, True,
                True,
                False,
                False,
            ],
            # fmt: on
        ),
        (
            [9],
            # fmt: off
            [
                True, True,
                True,
                True, True,
                True,
                True,
                True, True,
                True, True,
                True,
                True,
                False,
            ],
            # fmt: on
        ),
        (
            [10],
            # fmt: off
            [
                True, True,
                True,
                True, True,
                True,
                True,
                True, True,
                True, True,
                True,
                True,
                False,
            ],
            # fmt: on
        ),
    ],
)
def test__get_mask_invalid_actions_forward__returns_expected(env, state, mask_expected):
    assert env.get_mask_invalid_actions_forward(state) == mask_expected


@pytest.mark.parametrize(
    "state, action, next_state",
    [
        (
            [0],
            (0, 1),
            [1],
        ),
        (
            [0],
            (0, 2),
            [2],
        ),
        (
            [1],
            (1, 3),
            [3],
        ),
        (
            [2],
            (2, 3),
            [3],
        ),
        (
            [3],
            (3, 5),
            [5],
        ),
        (
            [4],
            (4, 6),
            [6],
        ),
        (
            [4],
            (-1, -1),
            [4],
        ),
    ],
)
def test__step__returns_expected(env, state, action, next_state):
    env.set_state(state)
    env.step(action)
    assert env.state == next_state


@pytest.mark.parametrize(
    "state, readable",
    [
        ([0], "s0"),
        ([1], "s1"),
        ([2], "s2"),
        ([3], "s3"),
        ([4], "s4"),
        ([5], "s5"),
        ([6], "s6"),
        ([7], "s7"),
        ([8], "s8"),
        ([9], "s9"),
        ([10], "s10"),
    ],
)
def test__state2readable__returns_expected(env, state, readable):
    assert env.state2readable(state) == readable


@pytest.mark.parametrize(
    "state, readable",
    [
        ([0], "s0"),
        ([1], "s1"),
        ([2], "s2"),
        ([3], "s3"),
        ([4], "s4"),
        ([5], "s5"),
        ([6], "s6"),
        ([7], "s7"),
        ([8], "s8"),
        ([9], "s9"),
        ([10], "s10"),
    ],
)
def test__readable2state__returns_expected(env, state, readable):
    assert env.readable2state(readable) == state


@pytest.mark.parametrize(
    "states",
    [[[3], [4], [6], [8], [9], [10]]],
)
def test__get_all_terminating_states__returns_expected(env, states):
    assert env.get_all_terminating_states() == states


@pytest.mark.parametrize(
    "states, s_proxy_expected",
    [
        (
            [[3], [4], [6], [8], [9], [10]],
            torch.tensor([[3], [4], [6], [8], [9], [10]], dtype=torch.long),
        )
    ],
)
def test__states2proxy__returns_expected(env, states, s_proxy_expected):
    assert torch.equal(env.states2proxy(states), s_proxy_expected)


@pytest.mark.parametrize(
    "states, s_policy_expected",
    [
        (
            [[3], [4], [6], [8], [9], [10]],
            torch.tensor(
                [
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                ],
                dtype=torch.long,
            ),
        ),
        (
            [[0], [0], [1], [1], [5], [10]],
            torch.tensor(
                [
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                ],
                dtype=torch.long,
            ),
        ),
    ],
)
def test__states2policy__returns_expected(env, states, s_policy_expected):
    assert torch.equal(env.states2policy(states), s_policy_expected)


class TestToyGenericCommon(common.BaseTestsDiscrete):
    """Common tests for the generic Toy environment."""

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
            "test__get_mask__is_consistent_regardless_of_inputs": 10,
            "test__get_valid_actions__is_consistent_regardless_of_inputs": 10,
            "test__sample_actions__get_logprobs__return_valid_actions_and_logprobs": 10,
            "test__get_parents_step_get_mask__are_compatible": 10,
            "test__sample_backwards_reaches_source": 10,
            "test__state2readable__is_reversible": 20,
            "test__gflownet_minimal_runs": 3,
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
        }
