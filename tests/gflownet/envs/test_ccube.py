import common
import numpy as np
import pytest
import torch
from torch.distributions import Bernoulli, Beta

from gflownet.envs.cube import ContinuousCube
from gflownet.utils.common import tbool, tfloat


@pytest.fixture
def cube1d():
    return ContinuousCube(n_dim=1, n_comp=3, min_incr=0.1, max_val=1.0)


@pytest.fixture
def cube2d():
    return ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1, max_val=1.0)


@pytest.mark.parametrize(
    "action_space",
    [
        [
            (0.0, 0.0),
            (-1.0, -1.0),
            (np.inf, np.inf),
        ],
    ],
)
@pytest.mark.skip(reason="skip while developping other tests")
def test__get_action_space__returns_expected(env, action_space):
    assert set(action_space) == set(env.action_space)


@pytest.mark.parametrize("env", ["cube1d", "cube2d"])
def test__get_policy_output__fixed_as_expected(env, request):
    env = request.getfixturevalue(env)
    policy_outputs = torch.unsqueeze(env.fixed_policy_output, 0)
    params = env.fixed_distr_params
    policy_output__as_expected(env, policy_outputs, params)


@pytest.mark.parametrize("env", ["cube1d", "cube2d"])
def test__get_policy_output__random_as_expected(env, request):
    env = request.getfixturevalue(env)
    policy_outputs = torch.unsqueeze(env.random_policy_output, 0)
    params = env.random_distr_params
    policy_output__as_expected(env, policy_outputs, params)


def policy_output__as_expected(env, policy_outputs, params):
    assert torch.all(
        env._get_policy_betas_weights(policy_outputs) == params["beta_weights"]
    )
    assert torch.all(
        env._get_policy_betas_alpha(policy_outputs) == params["beta_alpha"]
    )
    assert torch.all(env._get_policy_betas_beta(policy_outputs) == params["beta_beta"])
    assert torch.all(
        env._get_policy_eos_logit(policy_outputs) == params["bernoulli_eos_logit"]
    )
    assert torch.all(
        env._get_policy_source_logit(policy_outputs) == params["bernoulli_source_logit"]
    )


@pytest.mark.parametrize("env", ["cube1d", "cube2d"])
def test__mask_forward__returns_all_true_if_done(env, request):
    env = request.getfixturevalue(env)
    # Sample states
    states = env.get_uniform_terminating_states(100)
    # Iterate over state and test
    for state in states:
        env.set_state(state, done=True)
        mask = env.get_mask_invalid_actions_forward()
        assert all(mask)


@pytest.mark.parametrize("env", ["cube1d", "cube2d"])
def test__mask_backward__returns_all_true_except_eos_if_done(env, request):
    env = request.getfixturevalue(env)
    # Sample states
    states = env.get_uniform_terminating_states(100)
    # Iterate over state and test
    for state in states:
        env.set_state(state, done=True)
        mask = env.get_mask_invalid_actions_backward()
        assert all(mask[:-1])
        assert mask[-1] is False


@pytest.mark.parametrize(
    "state, mask_expected",
    [
        (
            [0.0],
            [False, False, True],
        ),
        (
            [0.5],
            [False, True, False],
        ),
        (
            [0.90],
            [False, True, False],
        ),
        (
            [0.95],
            [True, True, False],
        ),
    ],
)
def test__mask_forward__1d__returns_expected(cube1d, state, mask_expected):
    env = cube1d
    mask = env.get_mask_invalid_actions_forward(state)
    assert mask == mask_expected


@pytest.mark.parametrize(
    "state, mask_expected",
    [
        (
            [0.0, 0.0],
            [False, False, True],
        ),
        (
            [0.5, 0.5],
            [False, True, False],
        ),
        (
            [0.90, 0.5],
            [False, True, False],
        ),
        (
            [0.95, 0.5],
            [True, True, False],
        ),
        (
            [0.5, 0.90],
            [False, True, False],
        ),
        (
            [0.5, 0.95],
            [True, True, False],
        ),
        (
            [0.95, 0.95],
            [True, True, False],
        ),
    ],
)
def test__mask_forward__2d__returns_expected(cube2d, state, mask_expected):
    env = cube2d
    mask = env.get_mask_invalid_actions_forward(state)
    assert mask == mask_expected


@pytest.mark.parametrize(
    "state, mask_expected",
    [
        (
            [0.0],
            [True, False, True],
        ),
        (
            [0.1],
            [False, True, True],
        ),
        (
            [0.05],
            [True, False, True],
        ),
        (
            [0.5],
            [False, True, True],
        ),
        (
            [0.90],
            [False, True, True],
        ),
        (
            [0.95],
            [False, True, True],
        ),
    ],
)
def test__mask_backward__1d__returns_expected(cube1d, state, mask_expected):
    env = cube1d
    mask = env.get_mask_invalid_actions_backward(state)
    assert mask == mask_expected


@pytest.mark.parametrize(
    "state, mask_expected",
    [
        (
            [0.0, 0.0],
            [True, False, True],
        ),
        (
            [0.5, 0.5],
            [False, True, True],
        ),
        (
            [0.05, 0.5],
            [True, False, True],
        ),
        (
            [0.5, 0.05],
            [True, False, True],
        ),
        (
            [0.05, 0.05],
            [True, False, True],
        ),
        (
            [0.90, 0.5],
            [False, True, True],
        ),
        (
            [0.5, 0.90],
            [False, True, True],
        ),
        (
            [0.95, 0.5],
            [False, True, True],
        ),
        (
            [0.5, 0.95],
            [False, True, True],
        ),
        (
            [0.95, 0.95],
            [False, True, True],
        ),
    ],
)
def test__mask_backward__2d__returns_expected(cube2d, state, mask_expected):
    env = cube2d
    mask = env.get_mask_invalid_actions_backward(state)
    assert mask == mask_expected


@pytest.mark.parametrize(
    "state, increments_rel, min_increments, state_expected",
    [
        (
            [0.0, 0.0],
            [0.5, 0.5],
            [0.0, 0.0],
            [0.5, 0.5],
        ),
        (
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ),
        (
            [0.0, 0.0],
            [0.1794, 0.9589],
            [0.0, 0.0],
            [0.1794, 0.9589],
        ),
        (
            [0.3, 0.5],
            [0.0, 0.0],
            [0.1, 0.1],
            [0.4, 0.6],
        ),
        (
            [0.3, 0.5],
            [1.0, 1.0],
            [0.1, 0.1],
            [1.0, 1.0],
        ),
        (
            [0.3, 0.5],
            [0.5, 0.5],
            [0.1, 0.1],
            [0.7, 0.8],
        ),
        (
            [0.27, 0.85],
            [0.12, 0.76],
            [0.1, 0.1],
            [0.4456, 0.988],
        ),
        (
            [0.27, 0.95],
            [0.12, 0.0],
            [0.1, 0.0],
            [0.4456, 0.95],
        ),
        (
            [0.95, 0.27],
            [0.0, 0.12],
            [0.0, 0.1],
            [0.95, 0.4456],
        ),
    ],
)
def test__relative_to_absolute_increments__2d_forward__returns_expected(
    cube2d, state, increments_rel, min_increments, state_expected
):
    env = cube2d
    # Convert to tensors
    states = tfloat([state], float_type=env.float, device=env.device)
    increments_rel = tfloat([increments_rel], float_type=env.float, device=env.device)
    min_increments = tfloat([min_increments], float_type=env.float, device=env.device)
    states_expected = tfloat([state_expected], float_type=env.float, device=env.device)
    # Get absolute increments
    increments_abs = env.relative_to_absolute_increments(
        states, increments_rel, min_increments, env.max_val, is_backward=False
    )
    states_next = states + increments_abs
    assert torch.all(torch.isclose(states_next, states_expected))


@pytest.mark.parametrize(
    "state, increments_rel, min_increments, state_expected",
    [
        (
            [1.0, 1.0],
            [0.0, 0.0],
            [0.1, 0.1],
            [0.9, 0.9],
        ),
        (
            [1.0, 1.0],
            [1.0, 1.0],
            [0.1, 0.1],
            [0.0, 0.0],
        ),
        (
            [1.0, 1.0],
            [0.1794, 0.9589],
            [0.1, 0.1],
            [0.73854, 0.03699],
        ),
        (
            [0.3, 0.5],
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.4],
        ),
        (
            [0.3, 0.5],
            [1.0, 1.0],
            [0.1, 0.1],
            [0.0, 0.0],
        ),
    ],
)
def test__relative_to_absolute_increments__2d_backward__returns_expected(
    cube2d, state, increments_rel, min_increments, state_expected
):
    env = cube2d
    # Convert to tensors
    states = tfloat([state], float_type=env.float, device=env.device)
    increments_rel = tfloat([increments_rel], float_type=env.float, device=env.device)
    min_increments = tfloat([min_increments], float_type=env.float, device=env.device)
    states_expected = tfloat([state_expected], float_type=env.float, device=env.device)
    # Get absolute increments
    increments_abs = env.relative_to_absolute_increments(
        states, increments_rel, min_increments, env.max_val, is_backward=True
    )
    states_next = states - increments_abs
    assert torch.all(torch.isclose(states_next, states_expected))


@pytest.mark.parametrize(
    "state, action, state_expected",
    [
        (
            [0.0, 0.0],
            (0.5, 0.5),
            [0.5, 0.5],
        ),
        (
            [0.0, 0.0],
            (0.0, 0.0),
            [0.0, 0.0],
        ),
        (
            [0.0, 0.0],
            (0.1794, 0.9589),
            [0.1794, 0.9589],
        ),
        (
            [0.3, 0.5],
            (0.1, 0.1),
            [0.4, 0.6],
        ),
        (
            [0.3, 0.5],
            (0.7, 0.5),
            [1.0, 1.0],
        ),
        (
            [0.3, 0.5],
            (0.4, 0.3),
            [0.7, 0.8],
        ),
        (
            [0.27, 0.85],
            (0.1756, 0.138),
            [0.4456, 0.988],
        ),
        (
            [0.27, 0.95],
            (0.1756, 0.0),
            [0.4456, 0.95],
        ),
        (
            [0.95, 0.27],
            (0.0, 0.1756),
            [0.95, 0.4456],
        ),
    ],
)
def test__step_forward__2d__returns_expected(cube2d, state, action, state_expected):
    env = cube2d
    env.set_state(state)
    state_new, action, valid = env.step(action)
    assert env.isclose(state_new, state_expected)


@pytest.mark.parametrize(
    "states, force_eos",
    [
        (
            [[0.0, 0.0], [0.0, 0.0], [0.3, 0.5], [0.27, 0.85], [0.56, 0.23]],
            [False, False, False, False, False],
        ),
        (
            [[0.12, 0.17], [0.56, 0.23], [0.9, 0.9], [0.0, 0.0], [0.16, 0.93]],
            [False, False, False, False, False],
        ),
        (
            [[0.05, 0.97], [0.56, 0.23], [0.95, 0.3], [0.2, 0.95], [0.01, 0.01]],
            [False, False, False, False, False],
        ),
        (
            [[0.0, 0.0], [0.0, 0.0], [0.3, 0.5], [0.27, 0.85], [0.56, 0.23]],
            [False, False, False, True, False],
        ),
        (
            [[0.12, 0.17], [0.56, 0.23], [0.9, 0.9], [0.0, 0.0], [0.16, 0.93]],
            [False, True, True, False, False],
        ),
        (
            [[0.05, 0.97], [0.56, 0.23], [0.95, 0.98], [0.92, 0.95], [0.01, 0.01]],
            [False, False, False, True, True],
        ),
    ],
)
def test__sample_actions_forward__2d__returns_expected(cube2d, states, force_eos):
    env = cube2d
    n_states = len(states)
    force_eos = tbool(force_eos, device=env.device)
    # Get masks
    masks = tbool(
        [env.get_mask_invalid_actions_forward(s) for s in states], device=env.device
    )
    # Define Beta distribution with low variance and get confident range
    n_samples = 10000
    beta_params_min = 0.0
    beta_params_max = 10000
    alpha = 10
    alphas_presigmoid = alpha * torch.ones(n_samples)
    alphas = beta_params_max * torch.sigmoid(alphas_presigmoid) + beta_params_min
    beta = 1.0
    betas_presigmoid = beta * torch.ones(n_samples)
    betas = beta_params_max * torch.sigmoid(betas_presigmoid) + beta_params_min
    beta_distr = Beta(alphas, betas)
    samples = beta_distr.sample()
    mean_incr_rel = 0.9 * samples.mean()
    min_incr_rel = 0.9 * samples.min()
    max_incr_rel = 1.1 * samples.max()
    # Define Bernoulli parameters for EOS with deterministic probability
    logit_force_eos = torch.inf
    logit_force_noeos = -torch.inf
    # Estimate confident intervals of absolute actions
    states_torch = tfloat(states, float_type=env.float, device=env.device)
    is_source = torch.all(states_torch == 0.0, dim=1)
    is_near_edge = states_torch > 1.0 - env.min_incr
    min_increments = torch.full_like(
        states_torch, env.min_incr, dtype=env.float, device=env.device
    )
    min_increments[is_source, :] = 0.0
    increments_rel_min = torch.full_like(
        states_torch, min_incr_rel, dtype=env.float, device=env.device
    )
    increments_rel_max = torch.full_like(
        states_torch, max_incr_rel, dtype=env.float, device=env.device
    )
    increments_abs_min = env.relative_to_absolute_increments(
        states_torch, increments_rel_min, min_increments, env.max_val, is_backward=False
    )
    increments_abs_max = env.relative_to_absolute_increments(
        states_torch, increments_rel_max, min_increments, env.max_val, is_backward=False
    )
    # Get EOS actions
    is_eos_forced = torch.any(is_near_edge, dim=1)
    is_eos = torch.logical_or(is_eos_forced, force_eos)
    increments_abs_min[is_eos] = torch.inf
    increments_abs_max[is_eos] = torch.inf
    # Reconfigure environment
    env.n_comp = 1
    env.beta_params_min = 0.0
    env.beta_params_max = beta_params_max
    # Build policy outputs
    params = env.fixed_distr_params
    params["beta_alpha"] = alpha
    params["beta_beta"] = beta
    params["bernoulli_eos_logit"] = logit_force_noeos
    policy_outputs = torch.tile(env.get_policy_output(params), dims=(n_states, 1))
    policy_outputs[force_eos, -1] = logit_force_eos
    # Sample actions
    actions, _ = env.sample_actions_batch(
        policy_outputs, masks, states, is_backward=False
    )
    actions_tensor = tfloat(actions, float_type=env.float, device=env.device)
    actions_eos = torch.all(actions_tensor == torch.inf, dim=1)
    assert torch.all(actions_eos == is_eos)
    assert torch.all(actions_tensor >= increments_abs_min)
    assert torch.all(actions_tensor <= increments_abs_max)


@pytest.mark.parametrize(
    "states, force_bst",
    [
        (
            [[1.0, 1.0], [1.0, 1.0], [0.3, 0.5], [0.27, 0.85], [0.56, 0.23]],
            [False, False, False, False, False],
        ),
        (
            [[0.12, 0.17], [0.56, 0.23], [0.9, 0.9], [0.0, 0.05], [0.16, 0.93]],
            [False, False, False, False, False],
        ),
        (
            [[0.05, 0.97], [0.56, 0.23], [0.95, 0.3], [0.2, 0.95], [0.01, 0.01]],
            [False, False, False, False, False],
        ),
        (
            [[0.0001, 0.0], [0.001, 0.01], [0.3, 0.5], [0.27, 0.85], [0.56, 0.23]],
            [False, False, False, True, False],
        ),
        (
            [[0.12, 0.17], [0.56, 0.23], [0.9, 0.9], [1.0, 1.0], [0.16, 0.93]],
            [False, True, True, True, False],
        ),
        (
            [[0.05, 0.97], [0.56, 0.23], [0.95, 0.98], [0.92, 0.95], [0.01, 0.01]],
            [False, False, False, True, True],
        ),
    ],
)
def test__sample_actions_backward__2d__returns_expected(cube2d, states, force_bst):
    env = cube2d
    n_states = len(states)
    force_bst = tbool(force_bst, device=env.device)
    # Get masks
    masks = tbool(
        [env.get_mask_invalid_actions_backward(s) for s in states], device=env.device
    )
    # Define Beta distribution with low variance and get confident range
    n_samples = 10000
    beta_params_min = 0.0
    beta_params_max = 10000
    alpha = 10
    alphas_presigmoid = alpha * torch.ones(n_samples)
    alphas = beta_params_max * torch.sigmoid(alphas_presigmoid) + beta_params_min
    beta = 1.0
    betas_presigmoid = beta * torch.ones(n_samples)
    betas = beta_params_max * torch.sigmoid(betas_presigmoid) + beta_params_min
    beta_distr = Beta(alphas, betas)
    samples = beta_distr.sample()
    mean_incr_rel = 0.9 * samples.mean()
    min_incr_rel = 0.9 * samples.min()
    max_incr_rel = 1.1 * samples.max()
    # Define Bernoulli parameters for BST with deterministic probability
    logit_force_bst = torch.inf
    logit_force_nobst = -torch.inf
    # Estimate confident intervals of absolute actions
    states_torch = tfloat(states, float_type=env.float, device=env.device)
    is_near_edge = states_torch < env.min_incr
    min_increments = torch.full_like(
        states_torch, env.min_incr, dtype=env.float, device=env.device
    )
    increments_rel_min = torch.full_like(
        states_torch, min_incr_rel, dtype=env.float, device=env.device
    )
    increments_rel_max = torch.full_like(
        states_torch, max_incr_rel, dtype=env.float, device=env.device
    )
    increments_abs_min = env.relative_to_absolute_increments(
        states_torch, increments_rel_min, min_increments, env.max_val, is_backward=True
    )
    increments_abs_max = env.relative_to_absolute_increments(
        states_torch, increments_rel_max, min_increments, env.max_val, is_backward=True
    )
    # Get BST actions
    is_bst_forced = torch.any(is_near_edge, dim=1)
    is_bst = torch.logical_or(is_bst_forced, force_bst)
    increments_abs_min[is_bst] = states_torch[is_bst]
    increments_abs_max[is_bst] = states_torch[is_bst]
    # Reconfigure environment
    env.n_comp = 1
    env.beta_params_min = 0.0
    env.beta_params_max = beta_params_max
    # Build policy outputs
    params = env.fixed_distr_params
    params["beta_alpha"] = alpha
    params["beta_beta"] = beta
    params["bernoulli_source_logit"] = logit_force_nobst
    policy_outputs = torch.tile(env.get_policy_output(params), dims=(n_states, 1))
    policy_outputs[force_bst, -2] = logit_force_bst
    # Sample actions
    actions, _ = env.sample_actions_batch(
        policy_outputs, masks, states, is_backward=True
    )
    actions_tensor = tfloat(actions, float_type=env.float, device=env.device)
    actions_bst = torch.all(actions_tensor == states_torch, dim=1)
    assert torch.all(actions_bst == is_bst)
    assert torch.all(actions_tensor >= increments_abs_min)
    assert torch.all(actions_tensor <= increments_abs_max)


@pytest.mark.parametrize(
    "states, actions",
    [
        (
            [[0.95, 0.97], [0.96, 0.5], [0.5, 0.96]],
            [[0.02, 0.01], [0.01, 0.2], [0.3, 0.01]],
        ),
        (
            [[0.95, 0.97], [0.901, 0.5], [1.0, 1.0]],
            [[np.inf, np.inf], [0.01, 0.2], [0.3, 0.01]],
        ),
    ],
)
def test__get_logprobs_forward__2d__nearedge_returns_prob1(cube2d, states, actions):
    """
    The only valid action from 'near-edge' states is EOS, thus the the log probability
    should be zero, regardless of the action and the policy outputs
    """
    env = cube2d
    n_states = len(states)
    states_torch = tfloat(states, float_type=env.float, device=env.device)
    actions = tfloat(actions, float_type=env.float, device=env.device)
    # Get masks
    masks = tbool(
        [env.get_mask_invalid_actions_forward(s) for s in states], device=env.device
    )
    # Build policy outputs
    params = env.fixed_distr_params
    policy_outputs = torch.tile(env.get_policy_output(params), dims=(n_states, 1))
    # Add noise to policy outputs
    policy_outputs += torch.randn(policy_outputs.shape)
    # Get log probs
    logprobs = env.get_logprobs(
        policy_outputs, True, actions, states_torch, None, masks
    )
    assert torch.all(logprobs == 0.0)


@pytest.mark.parametrize(
    "states, actions",
    [
        (
            [[0.1, 0.2], [0.3, 0.5], [0.5, 0.95]],
            [[np.inf, np.inf], [np.inf, np.inf], [np.inf, np.inf]],
        ),
        (
            [[0.5, 0.97], [0.01, 0.01], [1.0, 1.0]],
            [[np.inf, np.inf], [np.inf, np.inf], [np.inf, np.inf]],
        ),
    ],
)
def test__get_logprobs_forward__2d__eos_actions_return_expected(
    cube2d, states, actions
):
    """
    The only valid action from 'near-edge' states is EOS, thus the the log probability
    should be zero, regardless of the action and the policy outputs
    """
    env = cube2d
    n_states = len(states)
    states_torch = tfloat(states, float_type=env.float, device=env.device)
    actions = tfloat(actions, float_type=env.float, device=env.device)
    # Get masks
    masks = tbool(
        [env.get_mask_invalid_actions_forward(s) for s in states], device=env.device
    )
    # Define Bernoulli parameter for EOS with deterministic probability (force EOS)
    # If Bernouilli has logit torch.inf, the logprobs are nan
    logit_force_eos = 1000
    # Build policy outputs
    params = env.fixed_distr_params
    params["bernoulli_eos_logit"] = logit_force_eos
    policy_outputs = torch.tile(env.get_policy_output(params), dims=(n_states, 1))
    # Add noise to policy outputs
    policy_outputs += torch.randn(policy_outputs.shape)
    # Get log probs
    logprobs = env.get_logprobs(
        policy_outputs, True, actions, states_torch, None, masks
    )
    assert torch.all(logprobs == 0.0)


@pytest.mark.parametrize(
    "state, expected",
    [
        (
            [0.0, 0.0],
            [0.0, 0.0],
        ),
        (
            [1.0, 1.0],
            [1.0, 1.0],
        ),
        (
            [1.1, 1.00001],
            [1.0, 1.0],
        ),
        (
            [-0.1, 1.00001],
            [0.0, 1.0],
        ),
        (
            [0.1, 0.21],
            [0.1, 0.21],
        ),
    ],
)
@pytest.mark.skip(reason="skip while developping other tests")
def test__state2policy_returns_expected(env, state, expected):
    assert env.state2policy(state) == expected


@pytest.mark.parametrize(
    "states, expected",
    [
        (
            [[0.0, 0.0], [1.0, 1.0], [1.1, 1.00001], [-0.1, 1.00001], [0.1, 0.21]],
            [[0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [0.0, 1.0], [0.1, 0.21]],
        ),
    ],
)
@pytest.mark.skip(reason="skip while developping other tests")
def test__statetorch2policy_returns_expected(env, states, expected):
    assert torch.equal(
        env.statetorch2policy(torch.tensor(states)), torch.tensor(expected)
    )


@pytest.mark.parametrize(
    "state, expected",
    [
        (
            [0.0, 0.0],
            [True, False, False],
        ),
        (
            [0.1, 0.1],
            [False, True, False],
        ),
        (
            [1.0, 0.0],
            [False, True, False],
        ),
        (
            [1.1, 0.0],
            [True, True, False],
        ),
        (
            [0.1, 1.1],
            [True, True, False],
        ),
    ],
)
@pytest.mark.skip(reason="skip while developping other tests")
def test__get_mask_invalid_actions_forward__returns_expected(env, state, expected):
    assert env.get_mask_invalid_actions_forward(state) == expected, print(
        state, expected, env.get_mask_invalid_actions_forward(state)
    )


@pytest.mark.skip(reason="skip while developping other tests")
def test__continuous_env_common__cube1d(cube1d):
    return common.test__continuous_env_common(cube1d)


def test__continuous_env_common__cube2d(cube2d):
    return common.test__continuous_env_common(cube2d)
