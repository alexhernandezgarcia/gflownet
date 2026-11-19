import math
from copy import copy

import common
import numpy as np
import pytest
import torch
from torch.distributions import Bernoulli
from utils_for_tests import load_base_test_config

from gflownet.envs.ctorus import ContinuousTorus
from gflownet.envs.nacctorus import NonAcyclicContinuousTorus
from gflownet.utils.batch import Batch
from gflownet.utils.common import gflownet_from_config

# --------------------------------------------------------------------------- #
# Fixtures, helpers
# --------------------------------------------------------------------------- #


@pytest.fixture
def env():
    """Default 2D environment with default fixed/random distr params."""
    return NonAcyclicContinuousTorus(n_dim=2, n_comp=3)


@pytest.fixture
def env_ctorus():
    return ContinuousTorus(n_dim=2, n_comp=3, length_traj=5, start_uniform=True)


@pytest.fixture
def env3d():
    return NonAcyclicContinuousTorus(n_dim=3, n_comp=2)


@pytest.fixture
def custom_params():
    fixed = {
        "vonmises_mean": 1.0,
        "vonmises_concentration": 2.0,
        "bernoulli_logit": 0.5,
    }
    random = {
        "vonmises_mean": -1.0,
        "vonmises_concentration": 0.01,
        "bernoulli_logit": -0.5,
    }
    return fixed, random


def make_gflownet(env, loss_name="tb"):
    if loss_name == "tb":
        overrides = ["gflownet=trajectorybalance", "loss=trajectorybalance"]
    elif loss_name == "vargrad":
        overrides = ["gflownet=vargrad", "loss=vargrad"]
    config = load_base_test_config(
        overrides=[
            "gflownet.optimizer.batch_size.forward=4",
            "gflownet.optimizer.n_train_steps=1",
            "buffer.replay_capacity=10",
        ]
        + overrides
    )

    # Initialize a GFlowNet agent from the configuration file
    gflownet = gflownet_from_config(config, env=env)

    return gflownet


def make_env(n_dim=2, n_comp=1):
    return NonAcyclicContinuousTorus(n_dim=n_dim, n_comp=n_comp)


# --------------------------------------------------------------------------- #
# __init__
# --------------------------------------------------------------------------- #


class TestInit:
    def test_default_params_are_set(self, env):
        assert env.fixed_distr_params["vonmises_mean"] == 0.0
        assert env.fixed_distr_params["vonmises_concentration"] == 0.5
        assert env.fixed_distr_params["bernoulli_logit"] == 0.0
        assert env.random_distr_params["vonmises_mean"] == 0.0
        assert env.random_distr_params["vonmises_concentration"] == 0.001
        assert env.random_distr_params["bernoulli_logit"] == 0.0

    # TODO: stopped here
    def test_custom_params_are_respected(self, custom_params):
        fixed, random = custom_params
        e = NonAcyclicContinuousTorus(
            n_dim=2,
            n_comp=1,
            fixed_distr_params=fixed,
            random_distr_params=random,
        )
        assert e.fixed_distr_params == fixed
        assert e.random_distr_params == random

    def test_length_traj_is_forced_to_one(self, env):
        # The subclass hardcodes length_traj=1 regardless of what a caller
        # might try to pass via kwargs to the parent.
        assert env.length_traj == 1

    def test_distr_type_is_von_mises(self, env):
        assert env.distr_type == "von_mises"

    def test_n_dim_propagated(self, env, env3d):
        assert env.n_dim == 2
        assert env3d.n_dim == 3

    def test_n_comp_propagated(self, env, env3d):
        assert env.n_comp == 3
        assert env3d.n_comp == 2

    def test_state_space_atol_default(self, env):
        assert env.state_space_atol == pytest.approx(1e-6)

    def test_state_space_atol_custom(self):
        e = NonAcyclicContinuousTorus(n_dim=2, state_space_atol=1e-3)
        assert e.state_space_atol == pytest.approx(1e-3)

    def test_vonmises_min_concentration_default(self, env):
        assert env.vonmises_min_concentration == pytest.approx(1e-3)

    def test_exp_vonmises_concentration_default_true(self, env):
        assert env.exp_vonmises_concentration is True

    def test_start_uniform_is_always_true(self, env):
        assert env.start_uniform is True


# --------------------------------------------------------------------------- #
# get_mask_invalid_actions_forward
# --------------------------------------------------------------------------- #


class TestMaskInvalidActionsForward:
    def test_done_true_returns_all_true(self, env):
        state = list(env.source)
        state[0] = 0.2  # not a source state anymore
        mask = env.get_mask_invalid_actions_forward(state=state, done=True)
        assert mask == [True, True]

    def test_source_state_returns_increment_valid_eos_invalid(self, env):
        state = list(env.source)
        mask = env.get_mask_invalid_actions_forward(state=state, done=False)
        assert mask == [False, True]

    def test_non_source_non_done_returns_both_valid(self, env):
        state = [0.1, 0.2]
        state[0] = 0.2  # not a source state anymore
        mask = env.get_mask_invalid_actions_forward(state=state, done=False)
        assert mask == [False, False]

    def test_uses_internal_state_and_done_when_none_passed(self, env):
        env.state = [0.1, 0.2]
        env.done = False
        mask = env.get_mask_invalid_actions_forward(state=None, done=None)
        assert mask == [False, False]

    def test_done_takes_priority_over_source(self, env):
        # Even if the state is the source, done=True must still fully mask.
        state = list(env.source)
        mask = env.get_mask_invalid_actions_forward(state=state, done=True)
        assert mask == [True, True]

    def test_return_type_is_list_of_bool(self, env):
        mask = env.get_mask_invalid_actions_forward(state=[0.1, 0.2], done=False)
        assert isinstance(mask, list)
        assert all(isinstance(m, bool) for m in mask)


# --------------------------------------------------------------------------- #
# get_mask_invalid_actions_backward
# --------------------------------------------------------------------------- #


class TestMaskInvalidActionsBackward:
    def test_done_true_returns_all_true(self, env):
        mask = env.get_mask_invalid_actions_backward(state=[0.1, 0.2], done=True)
        assert mask == [True, True]

    def test_done_false_returns_all_false(self, env):
        mask = env.get_mask_invalid_actions_backward(state=[0.1, 0.2], done=False)
        assert mask == [False, False]

    def test_uses_internal_state_and_done_when_none_passed(self, env):
        env.state = [0.1, 0.2]
        env.done = True
        mask = env.get_mask_invalid_actions_backward(state=None, done=None)
        assert mask == [True, True]

    def test_parents_a_is_ignored(self, env):
        # parents_a should have no effect on the mask value.
        mask_with = env.get_mask_invalid_actions_backward(
            state=[0.1, 0.2], done=False, parents_a=["anything"]
        )
        mask_without = env.get_mask_invalid_actions_backward(
            state=[0.1, 0.2], done=False, parents_a=None
        )
        assert mask_with == mask_without == [False, False]


# --------------------------------------------------------------------------- #
# get_valid_actions
# --------------------------------------------------------------------------- #


class TestGetValidActions:
    def test_forward_both_valid_returns_representative_and_eos(self, env):
        actions = env.get_valid_actions(mask=[False, False], backward=False)
        assert actions == [env.representative_action, env.eos]

    def test_forward_increment_invalid_returns_only_eos(self, env):
        actions = env.get_valid_actions(mask=[True, False], backward=False)
        assert actions == [env.eos]

    def test_forward_eos_invalid_returns_only_representative(self, env):
        actions = env.get_valid_actions(mask=[False, True], backward=False)
        assert actions == [env.representative_action]

    def test_forward_both_invalid_returns_empty(self, env):
        actions = env.get_valid_actions(mask=[True, True], backward=False)
        assert actions == []

    def test_backward_done_mask_returns_eos(self, env):
        actions = env.get_valid_actions(mask=[True, True], backward=True)
        assert actions == [env.eos]

    def test_backward_not_done_mask_returns_representative(self, env):
        actions = env.get_valid_actions(mask=[False, False], backward=True)
        assert actions == [env.representative_action]

    def test_mask_computed_internally_when_none_forward(self, env):
        actions = env.get_valid_actions(
            mask=None, state=[0.1, 0.2], done=False, backward=False
        )
        assert actions == [env.representative_action, env.eos]

    def test_mask_computed_internally_when_none_backward(self, env):
        actions = env.get_valid_actions(
            mask=None, state=[0.1, 0.2], done=True, backward=True
        )
        assert actions == [env.eos]


# --------------------------------------------------------------------------- #
# sample_actions_batch
# --------------------------------------------------------------------------- #


class TestSampleActionsBatch:
    def test_fist_action_increment_forward(self, env):
        policy_outputs = env.get_policy_output(env.fixed_distr_params).unsqueeze(0)
        mask = torch.tensor([env.get_mask_invalid_actions_forward()])
        states_from = [env.state]
        actions = env.sample_actions_batch(
            policy_outputs=policy_outputs,
            mask=mask,
            states_from=states_from,
            is_backward=False,
        )
        assert env.eos != actions[0]

    def test_later_actions_eos_or_increment_forward(self, env):
        env.state = [0.1, 0.2]
        env.done = False
        policy_outputs = env.get_policy_output(env.fixed_distr_params).unsqueeze(0)
        mask = torch.tensor([env.get_mask_invalid_actions_forward()])
        states_from = [env.state]
        encountered_eos = False
        encountered_increment = False
        for _ in range(50):
            actions = env.sample_actions_batch(
                policy_outputs=policy_outputs,
                mask=mask,
                states_from=states_from,
                is_backward=False,
            )
            if env.eos != actions[0]:
                encountered_increment = True
            else:
                encountered_eos = True
        assert encountered_eos and encountered_increment

    def test_first_action_eos_backward(self, env):
        env.state = [0.1, 0.2]
        env.done = True
        policy_outputs = env.get_policy_output(env.fixed_distr_params).unsqueeze(0)
        mask = torch.tensor([env.get_mask_invalid_actions_backward()])
        states_from = [env.state]
        actions = env.sample_actions_batch(
            policy_outputs=policy_outputs,
            mask=mask,
            states_from=states_from,
            is_backward=True,
        )
        assert env.eos == actions[0]

    def test_later_actions_only_increment_backward_with_back_to_source(self, env):
        env.state = [0.1, 0.2]
        env.done = False
        policy_outputs = env.get_policy_output(env.fixed_distr_params).unsqueeze(0)
        mask = torch.tensor([env.get_mask_invalid_actions_backward()])
        states_from = [env.state]
        encountered_eos = False
        encountered_increment = False
        encountered_bts = False
        for _ in range(50):
            actions = env.sample_actions_batch(
                policy_outputs=policy_outputs,
                mask=mask,
                states_from=states_from,
                is_backward=True,
            )
            if env.eos != actions[0]:
                encountered_increment = True
                if env.isclose(env.state, actions[0]):
                    encountered_bts = True
            else:
                encountered_eos = True
        assert not encountered_eos and encountered_increment and encountered_bts

    def test__fist_action_forward_is_uniform(self, env):
        n_states = 500
        policy_outputs = env.get_policy_output(env.fixed_distr_params).unsqueeze(0)
        policy_outputs = torch.cat([policy_outputs] * n_states, dim=0)
        assert env.is_source()
        states_from = [copy(env.source) for _ in range(n_states)]
        mask = torch.tensor([env.get_mask_invalid_actions_forward()] * n_states)

        actions = env.sample_actions_batch(
            policy_outputs=policy_outputs,
            mask=mask,
            states_from=states_from,
            is_backward=False,
        )
        actions = np.array(actions)

        # statistics should be close to the ones for Uniform[0, 2pi]
        expected_mean = np.pi
        expected_std = np.sqrt(1 / 12) * 2 * np.pi

        assert np.isclose(expected_mean, actions.mean(), atol=0.1)
        assert np.isclose(expected_std, actions.std(), atol=0.1)


# --------------------------------------------------------------------------- #
# get_logprobs
# --------------------------------------------------------------------------- #


class TestGetLogprobs:
    def test_increment_from_source_action_forward(self, env, env_ctorus):
        policy_outputs = env.get_policy_output(env.fixed_distr_params).unsqueeze(0)
        mask = torch.tensor([env.get_mask_invalid_actions_forward()])
        actions = [[0.1, 0.2]]
        states_from = [env.state]
        logprobs = env.get_logprobs(
            policy_outputs=policy_outputs,
            actions=actions,
            mask=mask,
            states_from=states_from,
            is_backward=False,
        )
        assert len(logprobs) == 1
        assert logprobs[0] < 0

        mask = torch.tensor([[False, False]])
        states_from = [env_ctorus.state]
        logprobs_ctorus = env_ctorus.get_logprobs(
            policy_outputs=policy_outputs[:, :-1],
            actions=actions,
            mask=mask,
            states_from=states_from,
            is_backward=False,
        )
        # logprobs should be the same as only the increment is possible from source
        assert logprobs == logprobs_ctorus

    def test_increment_action_not_from_source_forward(self, env):
        env.state = [0.1, 0.2]
        policy_outputs = env.get_policy_output(env.fixed_distr_params).unsqueeze(0)
        mask = torch.tensor([env.get_mask_invalid_actions_forward()])
        actions = [[0.1, 0.2]]
        states_from = [env.state]
        logprobs = env.get_logprobs(
            policy_outputs=policy_outputs,
            actions=actions,
            mask=mask,
            states_from=states_from,
            is_backward=False,
        )

        mask = torch.tensor([[False, False]])
        states_from = [env.state + [0]]
        logprobs_ctorus = ContinuousTorus.get_logprobs(
            self=env,
            policy_outputs=policy_outputs[:, :-1],
            actions=actions,
            mask=mask,
            states_from=states_from,
            is_backward=False,
        )
        # the nac logrpob should be smaller as EOS is also allowed fron non-source state
        assert logprobs < logprobs_ctorus

    def test_eos_action_not_from_source_forward(self, env):
        env.state = [0.1, 0.2]
        policy_outputs = env.get_policy_output(env.fixed_distr_params).unsqueeze(0)
        mask = torch.tensor([env.get_mask_invalid_actions_forward()])
        actions = [env.eos]
        states_from = [env.state]
        logprobs = env.get_logprobs(
            policy_outputs=policy_outputs,
            actions=actions,
            mask=mask,
            states_from=states_from,
            is_backward=False,
        )
        assert len(logprobs) == 1
        assert logprobs[0] < 0.0

    def test_increment_action_from_done_backward(self, env):
        actions = [[0.1, 0.2], [0.3, -0.5]]
        n_actions = len(actions)
        env.state = [0.1, 0.1]
        env.done = True
        policy_outputs = env.get_policy_output(env.fixed_distr_params).unsqueeze(0)
        policy_outputs = torch.cat([policy_outputs] * n_actions, dim=0)
        mask = torch.tensor([env.get_mask_invalid_actions_backward()] * n_actions)
        states_from = [env.state] * n_actions
        logprobs = env.get_logprobs(
            policy_outputs=policy_outputs,
            actions=actions,
            mask=mask,
            states_from=states_from,
            is_backward=True,
        )

        mask = torch.tensor([[False, True]] * n_actions)
        states_from = [env.state + [0]] * n_actions
        logprobs_ctorus = ContinuousTorus.get_logprobs(
            self=env,
            policy_outputs=policy_outputs[:, :-1],
            actions=actions,
            mask=mask,
            states_from=states_from,
            is_backward=True,
        )
        # TODO: not sure if we want it to be 0., this action is invalid and should not have lp=0.
        assert logprobs[0] == logprobs_ctorus[0] == 0.0

    def test_eos_action_from_done_backward(self, env):
        env.state = [0.1, 0.1]
        env.done = True
        policy_outputs = env.get_policy_output(env.fixed_distr_params).unsqueeze(0)
        mask = torch.tensor([env.get_mask_invalid_actions_backward()])
        actions = [env.done]
        states_from = [env.state]
        logprobs = env.get_logprobs(
            policy_outputs=policy_outputs,
            actions=actions,
            mask=mask,
            states_from=states_from,
            is_backward=True,
        )

        mask = torch.tensor([[False, True]])
        states_from = [env.state + [0]]
        logprobs_ctorus = ContinuousTorus.get_logprobs(
            self=env,
            policy_outputs=policy_outputs[:, :-1],
            actions=actions,
            mask=mask,
            states_from=states_from,
            is_backward=True,
        )
        assert logprobs == logprobs_ctorus == 0.0

    def test_increment_bts_action_backward(self, env):
        env.state = [0.1, 0.1]
        env.done = False
        policy_outputs = env.get_policy_output(env.fixed_distr_params).unsqueeze(0)
        policy_outputs = torch.cat([policy_outputs, policy_outputs], dim=0)
        mask = torch.tensor([env.get_mask_invalid_actions_backward()] * 2)
        # increment and back-to-source actions
        actions = [[0.2, 0.4], [0.1, 0.1]]
        states_from = [env.state, env.state]
        logprobs = env.get_logprobs(
            policy_outputs=policy_outputs,
            actions=actions,
            mask=mask,
            states_from=states_from,
            is_backward=True,
        )

        mask = torch.tensor([[False, False]] * 2)
        states_from = [env.state + [0]] * 2
        logprobs_ctorus = ContinuousTorus.get_logprobs(
            self=env,
            policy_outputs=policy_outputs[:, :-1],
            actions=actions,
            mask=mask,
            states_from=states_from,
            is_backward=True,
        )
        # logp(bts) has only discrete component where p = 1/2 according to
        # the fixed policy. Therefore, (1-p) = 1/2 and logp(incr) =
        # logp_continuous(incr) + log(1/2) = logp_continuous(incr) +
        # + logp(bts)
        assert logprobs[0] == logprobs_ctorus[0] + logprobs[1]


# --------------------------------------------------------------------------- #
# _step
# --------------------------------------------------------------------------- #


class TestStepInternal:
    def test_forward_step_adds_and_wraps(self, env):
        env.state = [0.1, 6.2]
        env._step((0.2, 0.2), backward=False)
        assert env.state[0] == pytest.approx(0.3)
        # 6.2 + 0.2 = 6.4 > 2*pi (~6.283) -> should wrap around
        assert env.state[1] == pytest.approx((6.2 + 0.2) % (2 * np.pi))

    def test_backward_step_subtracts_and_wraps(self, env):
        env.state = [0.1, 0.1]
        env._step((0.2, 0.5), backward=True)
        assert env.state[0] == pytest.approx((0.1 - 0.2) % (2 * np.pi))
        assert env.state[1] == pytest.approx((0.1 - 0.5) % (2 * np.pi))

    def test_result_always_in_0_2pi_range(self, env):
        env.state = [0.0, 0.0]
        env._step((-100.0, 100.0), backward=False)
        for angle in env.state:
            assert 0.0 <= angle < 2 * np.pi + 1e-9


# --------------------------------------------------------------------------- #
# step / step_backwards
# --------------------------------------------------------------------------- #


class TestStep:
    def test_step_valid_action_updates_state_from_source(self, env):
        new_state, action, valid = env.step((0.1, 0.2))
        assert valid is True
        assert new_state[0] == pytest.approx(0.1)
        assert new_state[1] == pytest.approx(0.2)

    def test_step_invalid_action_from_source(self, env):
        new_state, action, valid = env.step(env.eos)
        assert valid is False
        assert new_state == env.source

    def test_step_backwards_valid_action_updates_state(self, env):
        env.state = [0.5, 0.5]
        env.done = False
        new_state, action, valid = env.step_backwards((0.1, 0.2))
        assert valid is True
        assert new_state[0] == pytest.approx(0.4)
        assert new_state[1] == pytest.approx(0.3)

    def test_step_backwards_from_done_just_clears_done_flag(self, env):
        env.state = [0.5, 0.5]
        env.done = True
        new_state, action, valid = env.step_backwards(env.eos)
        assert valid is True
        assert env.done is False
        assert new_state == [0.5, 0.5]

    def test_step_backwards_invalid_action_returns_unchanged(self, env):
        env.state = [0.5, 0.5]
        env.done = False
        new_state, action, valid = env.step_backwards(env.eos)
        assert valid is False
        assert new_state == [0.5, 0.5]


# --------------------------------------------------------------------------- #
# get_grid_terminating_states / get_uniform_terminating_states
# --------------------------------------------------------------------------- #


class TestTerminatingStates:
    def test_pops_step_element_and_returns_n_states_grid(self, env):
        n = 10
        out = env.get_grid_terminating_states(n)
        assert len(out) >= n
        assert all(len(s) == 2 for s in out)  # step element removed

    def test_pops_step_element_and_returns_n_states_uniform(self, env):
        n = 10
        out = env.get_uniform_terminating_states(n)
        assert len(out) == n
        assert all(len(s) == 2 for s in out)  # step element removed


# --------------------------------------------------------------------------- #
# states2policy
# --------------------------------------------------------------------------- #
class TestStatesToPolicy:
    def test_states_to_policy_basic(self, env):
        states = [
            [0.1, 0.2],
            [0.3, 0.4],
            [0.3, 0.8],
            [0.3, 0.1],
        ]
        states_policy = env.states2policy(states)
        assert states_policy.shape[0] == len(states)
        assert states_policy.shape[1] == 2

        env.policy_encoding_dim_per_angle = 6
        states_policy = env.states2policy(states)
        assert states_policy.shape[0] == len(states)
        assert states_policy.shape[1] == 12


# copypasted from ctorus tests
class TestNonAcyclicContinuousTorusBasic(common.BaseTestsContinuous):
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
            "test__sample_backwards_reaches_source": 10,
            "test__state2readable__is_reversible": 20,
            "test__gflownet_minimal_runs": 3,
        }
        self.n_states = {
            "test__backward_actions_have_nonzero_forward_prob": 10,
            "test__sample_backwards_reaches_source": 10,
            "test__get_logprobs__all_finite_in_random_forward_transitions": 10,
            "test__get_logprobs__all_finite_in_random_backward_transitions": 10,
        }
        self.batch_size = {
            "test__sample_actions__get_logprobs__batched_forward_trajectories": 10,
            "test__sample_actions__get_logprobs__batched_backward_trajectories": 10,
            "test__get_logprobs__all_finite_in_accumulated_forward_trajectories": 10,
            "test__gflownet_minimal_runs": 10,
        }
        self.n_states = {}  # TODO: Populate.


class TestWithGFN:
    def test_basic_backward_sampling(self, env):
        gfn = make_gflownet(env)
        n_states = 4
        states_term = env.get_grid_terminating_states(n_states=n_states)

        logprobs_x_tt, logprobs_std, probs_std = gfn.estimate_logprobs_data(
            states_term,
            n_trajectories=2,
            max_data_size=100,
            batch_size=5,
            bs_num_samples=3,
        )
        assert torch.all(logprobs_x_tt < 0.0)

    def test__first_action_and_step_forward_is_uniform(self):
        n_states = 500
        n_dim = 2
        n_comp = 3
        envs = [make_env(n_dim, n_comp) for _ in range(n_states)]
        states_from = [env.state for env in envs]
        mask = torch.tensor([env.get_mask_invalid_actions_forward() for env in envs])

        env = envs[0]
        policy_outputs = env.get_policy_output(env.fixed_distr_params).unsqueeze(0)
        policy_outputs = torch.cat([policy_outputs] * n_states, dim=0)

        gfn = make_gflownet(env)

        batch = Batch(
            env=gfn.env,
            proxy=gfn.proxy,
            device=gfn.device,
            float_type=gfn.float,
        )

        actions, logprobs, _ = gfn.sample_actions(
            envs,
            batch,
            backward=False,
            no_random=True,
            compute_reversed_logprobs=False,
        )
        actions_np = np.array(actions)

        # statistics should be close to the ones for Uniform[0, 2pi]
        expected_mean = np.pi
        expected_std = np.sqrt(1 / 12) * 2 * np.pi

        assert np.isclose(expected_mean, actions_np.mean(), atol=0.1)
        assert np.isclose(expected_std, actions_np.std(), atol=0.1)

        envs, actions, valids = gfn.step(envs, actions)
        states_np = np.array([env.state for env in envs])
        # states should coinside with the actions after the first step from source
        assert np.isclose(states_np, actions_np).all()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
