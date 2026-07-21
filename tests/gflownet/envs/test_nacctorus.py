import math

import numpy as np
import pytest
import torch
from torch.distributions import Bernoulli

from gflownet.envs.nacctorus import NonAcyclicContinuousTorus
from gflownet.envs.ctorus import ContinuousTorus

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def env():
    """Default 2D environment with default fixed/random distr params."""
    return NonAcyclicContinuousTorus(n_dim=2, n_comp=1)


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
        assert env.n_comp == 1
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

    def test_start_uniform_default_false(self, env):
        assert env.start_uniform is False

    def test_start_uniform_true(self):
        e = NonAcyclicContinuousTorus(n_dim=2, start_uniform=True)
        assert e.start_uniform is True


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
# action_is_valid
# --------------------------------------------------------------------------- #


class TestActionIsValid:
    def test_eos_and_increment_are_valid_forward(self, env):
        state = [0.1, 0.2]
        done = False
        mask = env.get_mask_invalid_actions_forward(state=state, done=done)
        action_eos = env.eos
        action_incr = [0.01, -0.3]
        assert env.action_is_valid(action_eos, mask, state, done, backward=False)
        assert env.action_is_valid(action_incr, mask, state, done, backward=False)

    def test_eos_and_increment_are_valid_forward_computed_internally(self, env):
        env.state = [0.1, 0.2]
        env.done = False
        action_eos = env.eos
        action_incr = [0.01, -0.3]
        assert env.action_is_valid(action_eos, backward=False)
        assert env.action_is_valid(action_incr, backward=False)

    def test_eos_invalid_and_increment_valid_forward_computed_internally(self, env):
        action_eos = env.eos
        action_incr = [0.01, -0.3]
        assert not env.action_is_valid(action_eos, backward=False)
        assert env.action_is_valid(action_incr, backward=False)

    def test_all_invalid_forward_computed_internally(self, env):
        env.state = [0.1, 0.2]
        env.done = True
        action_eos = env.eos
        action_incr = [0.01, -0.3]
        assert not env.action_is_valid(action_eos, backward=False)
        assert not env.action_is_valid(action_incr, backward=False)

    def test_eos_valid_and_increment_invalid_backward(self, env):
        state = [0.1, 0.2]
        done = True
        mask = env.get_mask_invalid_actions_backward(state=state, done=done)
        action_eos = env.eos
        action_incr = [0.01, -0.3]
        assert env.action_is_valid(action_eos, mask, state, done, backward=True)
        assert not env.action_is_valid(action_incr, mask, state, done, backward=True)

    def test_eos_valid_and_increment_invalid_backward_computed_internally(self, env):
        env.state = [0.1, 0.2]
        env.done = True
        action_eos = env.eos
        action_incr = [0.01, -0.3]
        assert env.action_is_valid(action_eos, backward=True)
        assert not env.action_is_valid(action_incr, backward=True)

    def test_eos_vinalid_and_increment_valid_backward_computed_internally(self, env):
        env.state = [0.1, 0.2]
        env.done = False
        action_eos = env.eos
        action_incr = [0.01, -0.3]
        assert not env.action_is_valid(action_eos, backward=True)
        assert env.action_is_valid(action_incr, backward=True)


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


# --------------------------------------------------------------------------- #
# get_logprobs
# --------------------------------------------------------------------------- #


class TestGetLogprobs:
    def test_increment_from_source_action_forward(self, env):
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
        states_from = [env.state + [0]]
        logprobs_ctorus = ContinuousTorus.get_logprobs(
            self=env,
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

    def test_increment_from_source_action_forward_start_uniform(self, env):
        env.start_uniform = True
        actions = [[0.1, 0.2], [-0.2, 1.4], [1.0, -0.3]]
        n_actions = len(actions)
        policy_outputs = env.get_policy_output(env.fixed_distr_params).unsqueeze(0)
        policy_outputs = torch.cat([policy_outputs] * n_actions, dim=0)
        mask = torch.tensor([env.get_mask_invalid_actions_forward()] * n_actions)
        states_from = [env.state] * n_actions
        logprobs = env.get_logprobs(
            policy_outputs=policy_outputs,
            actions=actions,
            mask=mask,
            states_from=states_from,
            is_backward=False,
        )
        assert len(logprobs) == 3
        assert logprobs[0] < 0
        assert logprobs[0] == logprobs[1] == logprobs[2]

        mask = torch.tensor([[False, False]] * n_actions)
        states_from = [env.state + [0]] * n_actions
        logprobs_ctorus = ContinuousTorus.get_logprobs(
            self=env,
            policy_outputs=policy_outputs[:, :-1],
            actions=actions,
            mask=mask,
            states_from=states_from,
            is_backward=False,
        )
        # logprobs should be the same as only the increment is possible from source
        assert (logprobs == logprobs_ctorus).all()

    def test_increment_action_not_from_source_forward_start_uniform(self, env):
        env.state = [0.1, 0.2]
        env.start_uniform = True
        actions = [[0.1, 0.2], [-0.2, 1.4], [1.0, -0.3], env.eos]
        n_actions = len(actions)
        policy_outputs = env.get_policy_output(env.fixed_distr_params).unsqueeze(0)
        policy_outputs = torch.cat([policy_outputs] * n_actions, dim=0)
        mask = torch.tensor([env.get_mask_invalid_actions_forward()] * n_actions)
        states_from = [env.state] * n_actions
        logprobs = env.get_logprobs(
            policy_outputs=policy_outputs,
            actions=actions,
            mask=mask,
            states_from=states_from,
            is_backward=False,
        )
        assert len(logprobs) == n_actions
        assert logprobs[3] < 0
        # discrete action p = 1/2 for the fixed policy
        assert logprobs[3] == torch.log(torch.tensor(0.5))
        assert logprobs[0] != logprobs[1] != logprobs[2] != logprobs[3]

        mask = torch.tensor([[False, False]] * n_actions)
        states_from = [env.state + [0]] * n_actions
        logprobs_ctorus = ContinuousTorus.get_logprobs(
            self=env,
            policy_outputs=policy_outputs[:, :-1],
            actions=actions,
            mask=mask,
            states_from=states_from,
            is_backward=False,
        )

        assert logprobs[0] < logprobs_ctorus[0]
        assert logprobs[1] < logprobs_ctorus[1]
        assert logprobs[2] < logprobs_ctorus[2]

    def test_increment_action_from_done_backward_start_uniform(self, env):
        actions = [[0.1, 0.2], [0.3, -0.5]]
        env.start_uniform = True
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

    def test_eos_action_from_done_backward_start_uniform(self, env):
        env.start_uniform = True
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

    def test_increment_bts_action_backward_start_uniform(self, env):
        env.start_uniform = True
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


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
