import itertools
import re

import common
import pytest
import torch

from gflownet.envs.sequences.selfies import PAD_TOKEN, SELFIES_VOCAB_SMALL, Selfies
from gflownet.utils.common import tlong

SELFIES_VOCAB_TEST = ["[C]", "[O]", "[N]"]
SELFIES_TOKEN_PATTERN = re.compile(r"\[[^\]]+\]")


@pytest.fixture
def env():
    return Selfies(selfies_vocab=SELFIES_VOCAB_TEST, max_length=3, device="cpu")


@pytest.fixture
def env_large():
    return Selfies(
        selfies_vocab=SELFIES_VOCAB_TEST,
        min_length=2,
        max_length=5,
        device="cpu",
    )


@pytest.fixture
def env_default_vocab():
    return Selfies(max_length=2, device="cpu")


def _make_state(env, length: int, offset: int = 0):
    tokens = [((offset + idx) % env.n_tokens) + 1 for idx in range(length)]
    return tlong(
        tokens + [env.pad_idx] * (env.max_length - length),
        device=env.device,
    )


def _state_tokens(env, state):
    length = int(env._get_seq_length(state))
    return [env.idx2token[int(idx)] for idx in state[:length].tolist()]


def _make_readable(env, state):
    return " ".join(_state_tokens(env, state))


def _make_proxy(env, state):
    return "".join(_state_tokens(env, state))


def _all_readable_strings(env):
    return {
        " ".join(tokens)
        for length in range(env.min_length, env.max_length + 1)
        for tokens in itertools.product(env.selfies_vocab, repeat=length)
    }


def _all_proxy_strings(env):
    return {readable.replace(" ", "") for readable in _all_readable_strings(env)}


def _tokenize_selfies_string(selfies_string):
    tokens = SELFIES_TOKEN_PATTERN.findall(selfies_string)
    assert "".join(tokens) == selfies_string
    return tokens


def _assert_valid_proxy_string(env, proxy_string, expected_length=None):
    assert PAD_TOKEN not in proxy_string
    assert " " not in proxy_string
    tokens = _tokenize_selfies_string(proxy_string)
    if expected_length is not None:
        assert len(tokens) == expected_length
    assert all(token in env.selfies_vocab for token in tokens)


def _sample_random_state_at_max_length(env, max_attempts: int = 100):
    for _ in range(max_attempts):
        env.reset()
        state = env.state
        while not env.done and env._get_seq_length() < env.max_length:
            state, action, valid = env.step_random()
            assert valid
            assert action in env.action_space
        if env._get_seq_length(state) == env.max_length:
            return state
    raise RuntimeError("Could not sample a random trajectory that reached max length.")


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


def test__get_action_space__returns_expected(env, env_large):
    for current_env in (env, env_large):
        assert current_env.min_length > 0
        assert current_env.max_length >= current_env.min_length
        assert current_env.action_space == [
            (token_idx,) for token_idx in range(1, current_env.n_tokens + 1)
        ] + [current_env.eos]


def test__state2readable_and_readable2state__are_consistent(env, env_large):
    for current_env, max_offset in ((env, 2), (env_large, 1)):
        for length, offset in (
            (0, 0),
            (current_env.min_length, 0),
            (current_env.max_length, max_offset),
        ):
            state = _make_state(current_env, length, offset=offset)
            readable = current_env.state2readable(state)

            assert readable == _make_readable(current_env, state)
            assert torch.equal(current_env.readable2state(readable), state)


def test__states2proxy__returns_compact_selfies_strings(env, env_large):
    for current_env, max_offset in ((env, 2), (env_large, 1)):
        states = [
            _make_state(current_env, 0),
            _make_state(current_env, current_env.min_length),
            _make_state(current_env, current_env.max_length, offset=max_offset),
        ]
        states_tensor = tlong(
            [state.tolist() for state in states],
            device=current_env.device,
        )
        expected_proxy = [_make_proxy(current_env, state) for state in states]

        assert current_env.states2proxy(states_tensor) == expected_proxy
        assert (
            current_env.states2proxy([state.tolist() for state in states])
            == expected_proxy
        )

        for state, proxy_string in zip(states, expected_proxy):
            assert current_env.state2proxy(state) == [proxy_string]
            _assert_valid_proxy_string(
                current_env,
                proxy_string,
                expected_length=int(current_env._get_seq_length(state)),
            )


def test__get_all_terminating_states__returns_expected_formats(env):
    states = env.get_all_terminating_states()
    readable_states = {env.state2readable(state) for state in states}
    proxy_states = set(env.states2proxy(states))

    assert readable_states == _all_readable_strings(env)
    assert proxy_states == _all_proxy_strings(env)
    assert "[C]" in proxy_states
    assert "[C][O][N]" in proxy_states

    for proxy_string in proxy_states:
        _assert_valid_proxy_string(env, proxy_string)


def test__get_uniform_terminating_states__return_valid_selfies_proxy_strings(env_large):
    states = env_large.get_uniform_terminating_states(100, seed=0)
    readable_states = _all_readable_strings(env_large)
    proxy_states = _all_proxy_strings(env_large)

    for state, proxy_string in zip(states, env_large.states2proxy(states)):
        readable = env_large.state2readable(state)
        assert readable in readable_states
        assert proxy_string in proxy_states
        _assert_valid_proxy_string(
            env_large,
            proxy_string,
            expected_length=int(env_large._get_seq_length(state)),
        )


def test__step_random_to_max_length__produces_valid_selfies_proxy_string(env_large):
    for _ in range(20):
        state = _sample_random_state_at_max_length(env_large)
        proxy_string = env_large.states2proxy([state])[0]

        assert env_large._get_seq_length(state) == env_large.max_length
        assert env_large.state2proxy(state) == [proxy_string]
        _assert_valid_proxy_string(
            env_large,
            proxy_string,
            expected_length=env_large.max_length,
        )


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
