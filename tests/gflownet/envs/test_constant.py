import common
import numpy as np
import pytest
import torch

from gflownet.envs.constant import Constant


@pytest.fixture
def env_list():
    return Constant(state=[7.0, 3.14], device="cpu")


@pytest.fixture
def env_tensor():
    return Constant(state=torch.zeros(5), device="cpu")


@pytest.fixture
def env_array():
    return Constant(state=np.zeros(5), device="cpu")


@pytest.mark.parametrize(
    "env",
    ["env_list", "env_tensor", "env_array"],
)
def test__environment_initializes_properly(env, request):
    env = request.getfixturevalue(env)
    assert True


@pytest.mark.parametrize(
    "env",
    ["env_list", "env_tensor", "env_array"],
)
def test__get_action_space__contains_only_eos(env, request):
    env = request.getfixturevalue(env)
    assert env.get_action_space() == [env.eos]


@pytest.mark.parametrize(
    "env",
    ["env_list", "env_tensor"],
)
def test__done_mask_step__as_expected(env, request):
    env = request.getfixturevalue(env)
    env.reset()
    assert env.equal(env.state, env.source)
    assert not env.done
    assert env.get_valid_actions() == [env.eos]
    state, action, valid = env.step_random()
    assert action == env.eos
    assert valid
    assert env.equal(state, env.source)
    assert env.equal(env.state, env.source)
    assert env.done


class TestConstantListCommon(common.BaseTestsDiscrete):
    """Common tests for a Constant with a list."""

    @pytest.fixture(autouse=True)
    def setup(self, env_list):
        self.env = env_list
        self.repeats = {
            "test__reset__state_is_source": 10,
            "test__forward_actions_have_nonzero_backward_prob": 10,
            "test__backward_actions_have_nonzero_forward_prob": 10,
            "test__trajectories_are_reversible": 10,
            "test__step_random__does_not_sample_invalid_actions_forward": 10,
            "test__step_random__does_not_sample_invalid_actions_backward": 10,
            "test__sample_actions__get_logprobs__return_valid_actions_and_logprobs": 10,
            "test__get_logprobs__all_finite_in_random_backward_transitions": 0,
            "test__get_parents_step_get_mask__are_compatible": 10,
            "test__sample_backwards_reaches_source": 10,
            "test__state2readable__is_reversible": 0,
            "test__gflownet_minimal_runs": 3,
            "test__gflownet_minimal_runs": 3,
            "test__get_mask__is_consistent_regardless_of_inputs": 0,
            "test__get_valid_actions__is_consistent_regardless_of_inputs": 0,
            "test__dimensionality_policy_representation": 0,
        }
        self.n_states = {
            "test__backward_actions_have_nonzero_forward_prob": 1,
            "test__sample_backwards_reaches_source": 1,
            "test__get_logprobs__all_finite_in_random_forward_transitions": 1,
            "test__get_logprobs__all_finite_in_random_backward_transitions": 1,
            "test__set_state__sets_expected_state_and_creates_copy": 1,
        }
        self.batch_size = {
            "test__sample_actions__get_logprobs__batched_forward_trajectories": 2,
            "test__sample_actions__get_logprobs__batched_backward_trajectories": 2,
            "test__get_logprobs__all_finite_in_accumulated_forward_trajectories": 2,
            "test__gflownet_minimal_runs": 2,
        }


class TestConstantTensorCommon(common.BaseTestsDiscrete):
    """Common tests for a Constant with a tensor."""

    @pytest.fixture(autouse=True)
    def setup(self, env_tensor):
        self.env = env_tensor
        self.repeats = {
            "test__reset__state_is_source": 10,
            "test__forward_actions_have_nonzero_backward_prob": 10,
            "test__backward_actions_have_nonzero_forward_prob": 10,
            "test__trajectories_are_reversible": 10,
            "test__step_random__does_not_sample_invalid_actions_forward": 10,
            "test__step_random__does_not_sample_invalid_actions_backward": 10,
            "test__sample_actions__get_logprobs__return_valid_actions_and_logprobs": 10,
            "test__get_logprobs__all_finite_in_random_backward_transitions": 0,
            "test__get_parents_step_get_mask__are_compatible": 10,
            "test__sample_backwards_reaches_source": 10,
            "test__state2readable__is_reversible": 0,
            "test__gflownet_minimal_runs": 3,
            "test__get_mask__is_consistent_regardless_of_inputs": 0,
            "test__get_valid_actions__is_consistent_regardless_of_inputs": 0,
            "test__dimensionality_policy_representation": 0,
        }
        self.n_states = {
            "test__backward_actions_have_nonzero_forward_prob": 1,
            "test__sample_backwards_reaches_source": 1,
            "test__get_logprobs__all_finite_in_random_forward_transitions": 1,
            "test__get_logprobs__all_finite_in_random_backward_transitions": 1,
            "test__set_state__sets_expected_state_and_creates_copy": 1,
        }
        self.batch_size = {
            "test__sample_actions__get_logprobs__batched_forward_trajectories": 2,
            "test__sample_actions__get_logprobs__batched_backward_trajectories": 2,
            "test__get_logprobs__all_finite_in_accumulated_forward_trajectories": 2,
            "test__gflownet_minimal_runs": 2,
        }
