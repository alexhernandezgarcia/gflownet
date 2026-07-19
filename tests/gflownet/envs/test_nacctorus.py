import math

import numpy as np
import pytest
import torch
from torch.distributions import Bernoulli

from gflownet.envs.nacctorus import NonAcyclicContinuousTorus

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


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
