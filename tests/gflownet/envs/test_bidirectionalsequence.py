"""This is the tests for bidirectional sequences, covering:
1. right-only appending
2. left-only appending
3. bidirectional appending
in case:
1. action space
2. parents
3. step
4. mask
5. states

on envs:
1. default
2. min_length=1, max_length=3
3. min_length=3, max_length=3
"""

import common
import pytest
import torch

from gflownet.envs.sequences.bidirectional import BidirectionalSequence
from gflownet.utils.common import tlong


@pytest.fixture(params=["bidirectional", "right", "left"])
def direction(request):
    return request.param


@pytest.fixture(
    params=[{}, {"min_length": 1, "max_length": 3}, {"min_length": 3, "max_length": 3}],
    ids=["default-length", "len-1-3", "len-3-3"],
)
def env(request, direction):
    return BidirectionalSequence(
        tokens=(-2, -1, 0, 1, 2),
        pad_token=10,
        right_only=direction == "right",
        left_only=direction == "left",
        device="cpu",
        **request.param,
    )


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
    BidirectionalSequence(tokens=tokens, pad_token=100, device="cpu")
    assert True


def test__get_action_space__returns_expected(env):
    if env.right_only:
        directions = [1]
    elif env.left_only:
        directions = [0]
    else:
        directions = [0, 1]
    expected = [
        (direction, env.token2idx[token])
        for direction in directions
        for token in env.tokens
    ]
    expected.append(env.eos)

    assert set(expected) == set(env.action_space)


# Test cases:
# for states: "", "-1", "2 1", "0 0"
@pytest.mark.parametrize("readable", ["", "-1", "2 1", "0 0"])
def test__get_parents__returns_expected(env, readable):
    state = env.readable2state(readable)

    tokens = readable.split()
    expected_parents = []
    expected_actions = []

    if tokens:
        if len(tokens) == 1:
            side = 0 if env.left_only else 1
            expected_actions.append((side, env.token2idx[int(tokens[0])]))
            expected_parents.append(env.readable2state(""))
            # expected_actions.append((1, env.token2idx[int(tokens[0])]))
            # in our setting, at source only right appending is allowed if bidirectional
        else:
            expected_parents_left = env.readable2state(" ".join(tokens[1:]))
            expected_parents_right = env.readable2state(" ".join(tokens[:-1]))

            expected_action_left = (0, env.token2idx[int(tokens[0])])
            expected_action_right = (1, env.token2idx[int(tokens[-1])])

            if env.right_only or not env.left_only:
                expected_parents.append(expected_parents_right)
                expected_actions.append(expected_action_right)

            if env.left_only or not env.right_only:
                expected_parents.append(expected_parents_left)
                expected_actions.append(expected_action_left)

    parents, parents_a = env.get_parents(state)
    assert len(parents) == len(expected_parents)
    assert len(parents_a) == len(expected_actions)
    for p, p_e in zip(parents, expected_parents):
        assert torch.equal(p, p_e)
    for p_a, p_a_e in zip(parents_a, expected_actions):
        assert p_a == p_a_e
        assert isinstance(p_a[0], int)


# Cover tests:
# 1. source + right 1/-2
# 2. source + left 1/-2
# 3. -2 + right 3/0
@pytest.mark.parametrize(
    "readable, action, expected",
    [
        ("", (1, 1), "-2"),
        ("", (0, 1), "-2"),
        ("-2", (1, 3), "-2 0"),
        ("-2", (0, 3), "0 -2"),
        ("-2 0", (1, 2), "-2 0 -1"),
        ("-2 0", (0, 2), "-1 -2 0"),
    ],
)
def test__step__returns_expected(env, readable, action, expected):
    if action not in env.action_space:
        pytest.skip("Direction is disabled")

    env.set_state(env.readable2state(readable), done=False)

    mask = env.get_mask_invalid_actions_forward()
    if mask[env.action_space.index(action)]:
        pytest.skip("Action is masked at this state")

    state, returned_action, valid = env.step(action)

    assert valid
    assert returned_action == action
    assert torch.equal(state, env.readable2state(expected))
    assert not env.done


def test__get_mask_forward__returns_expected(env):
    for length in range(env.max_length + 1):
        state = env.readable2state(" ".join(["-2"] * length))
        env.set_state(state, done=False)

        expected = []
        for action in env.action_space:
            if action == env.eos:
                invalid = length < env.min_length
            else:
                invalid = length == env.max_length
                if (
                    length == 0
                    and not env.right_only
                    and not env.left_only
                    and action[0] == 0
                ):
                    invalid = True
            expected.append(invalid)

        assert env.get_mask_invalid_actions_forward() == expected


@pytest.mark.parametrize(
    "states, states_policy",
    [
        (
            [[], [1, 3, 2]],
            [
                [],
                # fmt: off
                [
                    0, 1, 0, 0, 0, 0,
                    0, 0, 0, 1, 0, 0,
                    0, 0, 1, 0, 0, 0,
                ],
                # fmt: on
            ],
        ),
        (
            [[1, 3, 2]],
            [
                # fmt: off
                [
                    0, 1, 0, 0, 0, 0,
                    0, 0, 0, 1, 0, 0,
                    0, 0, 1, 0, 0, 0,
                ],
                # fmt: on
            ],
        ),
    ],
)
def test__states2policy__returns_expected(env, states, states_policy):
    padding_policy = [1, 0, 0, 0, 0, 0]
    padded_states = []
    padded_policy = []

    for i in range(len(states)):
        remaining = env.max_length - len(states[i])
        padded_states.append(states[i] + [0] * remaining)
        padded_policy.append(states_policy[i] + padding_policy * remaining)

    states = tlong(padded_states, device=env.device)
    states_policy = tlong(padded_policy, device=env.device)
    assert torch.equal(states_policy, env.states2policy(states))


@pytest.mark.parametrize(
    "states, states_proxy",
    [
        (
            [[], [1, 3, 2]],
            [[], [-2, 0, -1]],
        ),
        (
            [[1, 3, 2]],
            [[-2, 0, -1]],
        ),
    ],
)
def test__states2proxy__returns_expected(env, states, states_proxy):
    padded_states = []
    padded_proxy = []

    for i in range(len(states)):
        remaining = env.max_length - len(states[i])
        padded_states.append(states[i] + [0] * remaining)
        padded_proxy.append(states_proxy[i] + [10] * remaining)

    states = tlong(padded_states, device=env.device)
    assert env.states2proxy(states) == padded_proxy


@pytest.mark.parametrize(
    "state, readable",
    [
        ([], ""),
        ([1, 3, 2], "-2 0 -1"),
        ([1, 1, 4], "-2 -2 1"),
    ],
)
def test__state2readable__returns_expected(env, state, readable):
    state = state + [0] * (env.max_length - len(state))
    state = tlong(state, device=env.device)
    assert env.state2readable(state) == readable


@pytest.mark.parametrize(
    "state, readable",
    [
        ([], ""),
        ([1, 3, 2], "-2 0 -1"),
        ([1, 1, 4], "-2 -2 1"),
    ],
)
def test__readable2state__returns_expected(env, state, readable):
    state = state + [0] * (env.max_length - len(state))
    state = tlong(state, device=env.device)
    assert torch.equal(env.readable2state(readable), state)


def test__readable2state__empty_returns_copy_of_source(env):
    state = env.readable2state("")
    state[0] = 1

    expected = tlong([0] * env.max_length, device=env.device)
    assert torch.equal(env.source, expected)
    assert torch.equal(env.reset().state, env.source)


def test__get_uniform_terminating_states__returns_valid_states(env):
    states = env.get_uniform_terminating_states(100, seed=0)
    assert len(states) == 100
    for state in states:
        assert len(state) == env.max_length
        length = int(env._get_seq_length(state))
        assert env.min_length <= length <= env.max_length
        assert torch.all(state[:length] != env.pad_idx)
        assert torch.all(state[length:] == env.pad_idx)


def test__get_uniform_terminating_states__is_reproducible_with_seed(env):
    states_seed_0_a = env.get_uniform_terminating_states(25, seed=0)
    states_seed_0_b = env.get_uniform_terminating_states(25, seed=0)
    states_seed_1 = env.get_uniform_terminating_states(25, seed=1)
    assert all(
        torch.equal(state_a, state_b)
        for state_a, state_b in zip(states_seed_0_a, states_seed_0_b)
    )
    assert any(
        not torch.equal(state_a, state_b)
        for state_a, state_b in zip(states_seed_0_a, states_seed_1)
    )


def test__get_uniform_terminating_states__samples_lengths_uniformly_over_states():
    env = BidirectionalSequence(tokens=(0, 1), min_length=1, max_length=3, device="cpu")
    states = env.get_uniform_terminating_states(7000, seed=0)
    lengths = torch.tensor([int(env._get_seq_length(state)) for state in states])
    counts = torch.bincount(lengths, minlength=env.max_length + 1)[env.min_length :]
    observed = counts.to(torch.float64) / counts.sum()
    expected = torch.tensor([2**length for length in range(1, 4)], dtype=torch.float64)
    expected /= expected.sum()
    assert torch.allclose(observed, expected, atol=0.03)


class TestBidirectionalSequence(common.BaseTestsDiscrete):
    """Common tests for all BidirectionalSequence configurations"""

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
