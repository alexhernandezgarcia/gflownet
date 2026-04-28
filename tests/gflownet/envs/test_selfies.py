import itertools

import common
import pytest
import torch

from gflownet.envs.sequences.selfies import PAD_TOKEN, SELFIES_VOCAB_SMALL, SelfiesEnv
from gflownet.utils.common import tlong

SELFIES_VOCAB_TEST = ["[C]", "[O]", "[N]"]


@pytest.fixture
def env():
    return SelfiesEnv(selfies_vocab=SELFIES_VOCAB_TEST, max_length=3, device="cpu")


@pytest.fixture
def env_large():
    return SelfiesEnv(
        selfies_vocab=SELFIES_VOCAB_TEST,
        min_length=2,
        max_length=5,
        device="cpu",
    )


@pytest.fixture
def env_default_vocab():
    return SelfiesEnv(max_length=2, device="cpu")


def _make_state(env, length: int, offset: int = 0):
    tokens = [((offset + idx) % env.n_tokens) + 1 for idx in range(length)]
    return tlong(
        tokens + [env.pad_idx] * (env.max_length - length),
        device=env.device,
    )


def _make_readable(env, state):
    length = int(env._get_seq_length(state))
    return " ".join(env.idx2token[int(idx)] for idx in state[:length].tolist())


def _make_proxy(env, state):
    length = int(env._get_seq_length(state))
    return [env.idx2token[int(idx)] for idx in state[:length].tolist()] + [
        env.pad_token
    ] * (env.max_length - length)


def _all_readable_strings(env):
    return {
        " ".join(tokens)
        for length in range(env.min_length, env.max_length + 1)
        for tokens in itertools.product(env.selfies_vocab, repeat=length)
    }


def test__environment_initializes_with_default_vocab(env_default_vocab):
    assert env_default_vocab.selfies_vocab == SELFIES_VOCAB_SMALL
    assert env_default_vocab.tokens == tuple(SELFIES_VOCAB_SMALL)
    assert env_default_vocab.pad_token == PAD_TOKEN


def test__environment_initializes_with_custom_vocab(env_large):
    assert env_large.selfies_vocab == SELFIES_VOCAB_TEST
    assert env_large.tokens == tuple(SELFIES_VOCAB_TEST)
    assert env_large.pad_token == PAD_TOKEN
    assert env_large.min_length == 2
    assert env_large.max_length == 5


def test__get_action_space__returns_expected(env):
    assert env.min_length == 1
    assert env.max_length == 3
    assert env.action_space == [(1,), (2,), (3,), env.eos]


def test__get_action_space__returns_expected_large(env_large):
    assert env_large.min_length == 2
    assert env_large.max_length == 5
    assert env_large.action_space == [(1,), (2,), (3,), env_large.eos]


def test__states2proxy__returns_expected(env):
    assert env.min_length == 1
    assert env.max_length == 3
    empty_state = _make_state(env, 0)
    min_state = _make_state(env, env.min_length)
    max_state = _make_state(env, env.max_length, offset=2)
    states = tlong(
        [empty_state.tolist(), min_state.tolist(), max_state.tolist()],
        device=env.device,
    )
    assert env.states2proxy(states) == [
        [PAD_TOKEN] * env.max_length,
        _make_proxy(env, min_state),
        _make_proxy(env, max_state),
    ]


def test__states2proxy__returns_expected_large(env_large):
    assert env_large.min_length == 2
    assert env_large.max_length == 5
    empty_state = _make_state(env_large, 0)
    min_state = _make_state(env_large, env_large.min_length)
    max_state = _make_state(env_large, env_large.max_length, offset=1)
    states = tlong(
        [empty_state.tolist(), min_state.tolist(), max_state.tolist()],
        device=env_large.device,
    )
    assert env_large.states2proxy(states) == [
        [PAD_TOKEN] * env_large.max_length,
        _make_proxy(env_large, min_state),
        _make_proxy(env_large, max_state),
    ]


def test__state2readable__returns_expected(env):
    assert env.min_length == 1
    assert env.max_length == 3
    empty_state = _make_state(env, 0)
    min_state = _make_state(env, env.min_length)
    max_state = _make_state(env, env.max_length, offset=2)
    assert env.state2readable(empty_state) == ""
    assert env.state2readable(min_state) == _make_readable(env, min_state)
    assert env.state2readable(max_state) == _make_readable(env, max_state)


def test__state2readable__returns_expected_large(env_large):
    assert env_large.min_length == 2
    assert env_large.max_length == 5
    empty_state = _make_state(env_large, 0)
    min_state = _make_state(env_large, env_large.min_length)
    max_state = _make_state(env_large, env_large.max_length, offset=1)
    assert env_large.state2readable(empty_state) == ""
    assert env_large.state2readable(min_state) == _make_readable(env_large, min_state)
    assert env_large.state2readable(max_state) == _make_readable(env_large, max_state)


def test__readable2state__returns_expected(env):
    assert env.min_length == 1
    assert env.max_length == 3
    empty_state = _make_state(env, 0)
    min_state = _make_state(env, env.min_length)
    max_state = _make_state(env, env.max_length, offset=2)
    assert torch.equal(env.readable2state(""), empty_state)
    assert torch.equal(env.readable2state(_make_readable(env, min_state)), min_state)
    assert torch.equal(env.readable2state(_make_readable(env, max_state)), max_state)


def test__readable2state__returns_expected_large(env_large):
    assert env_large.min_length == 2
    assert env_large.max_length == 5
    empty_state = _make_state(env_large, 0)
    min_state = _make_state(env_large, env_large.min_length)
    max_state = _make_state(env_large, env_large.max_length, offset=1)
    assert torch.equal(env_large.readable2state(""), empty_state)
    assert torch.equal(
        env_large.readable2state(_make_readable(env_large, min_state)), min_state
    )
    assert torch.equal(
        env_large.readable2state(_make_readable(env_large, max_state)), max_state
    )


def test__get_all_terminating_states__returns_expected_readable_strings(env):
    readable_states = {
        env.state2readable(state) for state in env.get_all_terminating_states()
    }
    assert readable_states == _all_readable_strings(env)
    assert "[C]" in readable_states
    assert "[C] [O] [N]" in readable_states


def test__get_uniform_terminating_states__return_valid_selfies_strings(env_large):
    states = env_large.get_uniform_terminating_states(100, seed=0)
    all_readable_states = _all_readable_strings(env_large)
    for state in states:
        readable = env_large.state2readable(state)
        assert readable in all_readable_states
        tokens = readable.split(" ")
        assert env_large.min_length <= len(tokens) <= env_large.max_length
        assert all(token in env_large.selfies_vocab for token in tokens)


class TestSelfiesSmallVocabCommonDiscrete(common.BaseTestsDiscrete):
    """Run the shared discrete-env test suite on a small SELFIES env."""

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
        }
        self.batch_size = {
            "test__sample_actions__get_logprobs__batched_forward_trajectories": 10,
            "test__sample_actions__get_logprobs__batched_backward_trajectories": 10,
            "test__get_logprobs__all_finite_in_accumulated_forward_trajectories": 10,
        }


class TestSelfiesLargeEnvCommonDiscrete(common.BaseTestsDiscrete):
    """Run the shared discrete-env test suite on a larger SELFIES env."""

    @pytest.fixture(autouse=True)
    def setup(self, env_large):
        self.env = env_large
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
