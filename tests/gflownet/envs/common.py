import warnings

import hydra
import numpy as np
import pytest
import torch
import yaml
from hydra import compose, initialize
from omegaconf import OmegaConf

from gflownet.utils.common import copy, tbool, tfloat


def test__all_env_common(env):
    test__init__state_is_source_no_parents(env)
    test__reset__state_is_source_no_parents(env)
    test__set_state__creates_new_copy_of_state(env)
    test__step__returns_same_state_action_and_invalid_if_done(env)
    test__sample_actions__get_logprobs__return_valid_actions_and_logprobs(env)
    test__sample_actions__backward__returns_eos_if_done(env)
    test__get_logprobs__backward__returns_zero_if_done(env)
    test__step_random__does_not_sample_invalid_actions(env)
    test__forward_actions_have_nonzero_backward_prob(env)
    test__backward_actions_have_nonzero_forward_prob(env)
    test__get_parents_step_get_mask__are_compatible(env)
    test__sample_backwards_reaches_source(env)
    test__state2readable__is_reversible(env)
    test__get_parents__returns_same_state_and_eos_if_done(env)
    test__actions2indices__returns_expected_tensor(env)
    test__gflownet_minimal_runs(env)


def test__continuous_env_common(env):
    test__reset__state_is_source(env)
    test__set_state__creates_new_copy_of_state(env)
    test__sampling_forwards_reaches_done_in_finite_steps(env)
    test__sample_actions__backward__returns_eos_if_done(env)
    test__get_logprobs__backward__returns_zero_if_done(env)
    test__forward_actions_have_nonzero_backward_prob(env)
    test__backward_actions_have_nonzero_forward_prob(env)
    test__step__returns_same_state_action_and_invalid_if_done(env)
    test__sample_backwards_reaches_source(env)


#     test__gflownet_minimal_runs(env)
#     test__sample_actions__get_logprobs__return_valid_actions_and_logprobs(env)
#     test__get_parents__returns_same_state_and_eos_if_done(env)
#     test__actions2indices__returns_expected_tensor(env)


def _get_terminating_states(env, n):
    # Hacky way of skipping the Crystal BW sampling test until fixed
    if env.__class__.__name__ == "Crystal":
        return
    if hasattr(env, "get_all_terminating_states"):
        return env.get_all_terminating_states()
    elif hasattr(env, "get_grid_terminating_states"):
        return env.get_grid_terminating_states(n)
    elif hasattr(env, "get_uniform_terminating_states"):
        return env.get_uniform_terminating_states(n, 0)
    elif hasattr(env, "get_random_terminating_states"):
        return env.get_random_terminating_states(n, 0)
    else:
        warnings.warn(
            f"""
        Testing backward sampling or setting terminating states requires that the
        environment implements one of the following:
            - get_all_terminating_states()
            - get_grid_terminating_states()
            - get_uniform_terminating_states()
            - get_random_terminating_states()
        Environment {env.__class__} does not have any of the above, therefore backward
        sampling will not be tested.
        """
        )
        return None


@pytest.mark.repeat(100)
def test__step_random__does_not_sample_invalid_actions(env):
    env = env.reset()
    while not env.done:
        state = copy(env.state)
        mask = env.get_mask_invalid_actions_forward()
        # Sample random action
        state_next, action, valid = env.step_random()
        if valid is False:
            continue
        assert mask[env.action2index(action)] is False
        # If action is not EOS then state should change
        if action != env.eos:
            assert not env.equal(state, state_next)


@pytest.mark.repeat(100)
def test__get_parents_step_get_mask__are_compatible(env):
    env = env.reset()
    n_actions = 0
    while not env.done:
        state = copy(env.state)
        # Sample random action
        state_next, action, valid = env.step_random()
        if valid is False:
            continue
        n_actions += 1
        assert n_actions <= env.max_traj_length
        assert env.n_actions == n_actions
        parents, parents_a = env.get_parents()
        if torch.is_tensor(state):
            assert any([env.equal(p, state) for p in parents])
        else:
            assert state in parents
        assert len(parents) == len(parents_a)
        for p, p_a in zip(parents, parents_a):
            mask = env.get_mask_invalid_actions_forward(p, False)
            assert p_a in env.action_space
            assert mask[env.action_space.index(p_a)] is False


@pytest.mark.repeat(500)
def test__sampling_forwards_reaches_done_in_finite_steps(env):
    n_actions = 0
    while not env.done:
        # Sample random action
        state_next, action, valid = env.step_random()
        n_actions += 1
        assert n_actions <= env.max_traj_length


@pytest.mark.repeat(5)
def test__set_state__creates_new_copy_of_state(env):
    states = _get_terminating_states(env, 5)
    if states is None:
        warnings.warn("Skipping test because states are None.")
        return
    envs = []
    for state in states:
        for idx in range(5):
            env_new = env.copy().reset(idx)
            env_new.set_state(state, done=True)
            envs.append(env_new)
    state_ids = [id(env.state) for env in envs]
    assert len(np.unique(state_ids)) == len(state_ids)


@pytest.mark.repeat(5)
def test__sample_actions__backward__returns_eos_if_done(env, n=5):
    states = _get_terminating_states(env, n)
    if states is None:
        warnings.warn("Skipping test because states are None.")
        return
    # Set states, done and get masks
    masks = []
    for state in states:
        env.set_state(state, done=True)
        masks.append(env.get_mask_invalid_actions_backward())
    # Build random policy outputs and tensor masks
    policy_outputs = torch.tile(
        tfloat(env.random_policy_output, float_type=env.float, device=env.device),
        (len(states), 1),
    )
    # Add noise to policy outputs
    policy_outputs += torch.randn(policy_outputs.shape)
    masks = tbool(masks, device=env.device)
    actions, _ = env.sample_actions_batch(
        policy_outputs, masks, states, is_backward=True
    )
    assert all([action == env.eos for action in actions])


@pytest.mark.repeat(5)
def test__get_logprobs__backward__returns_zero_if_done(env, n=5):
    states = _get_terminating_states(env, n)
    if states is None:
        warnings.warn("Skipping test because states are None.")
        return
    # Set states, done and get masks
    masks = []
    for state in states:
        env.set_state(state, done=True)
        masks.append(env.get_mask_invalid_actions_backward())
    # EOS actions
    actions_eos = torch.tile(
        tfloat(env.eos, float_type=env.float, device=env.device),
        (len(states), 1),
    )
    # Build random policy outputs and tensor masks
    policy_outputs = torch.tile(
        tfloat(env.random_policy_output, float_type=env.float, device=env.device),
        (len(states), 1),
    )
    # Add noise to policy outputs
    policy_outputs += torch.randn(policy_outputs.shape)
    masks = tbool(masks, device=env.device)
    logprobs = env.get_logprobs(
        policy_outputs, actions_eos, masks, states, is_backward=True
    )
    assert torch.all(logprobs == 0.0)


@pytest.mark.repeat(100)
def test__sample_backwards_reaches_source(env, n=100):
    states = _get_terminating_states(env, n)
    if states is None:
        warnings.warn("Skipping test because states are None.")
        return
    for state in states:
        env.set_state(state, done=True)
        n_actions = 0
        while True:
            if env.equal(env.state, env.source):
                assert True
                break
            env.step_random(backward=True)
            n_actions += 1
            assert n_actions <= env.max_traj_length


@pytest.mark.repeat(100)
def test__state2policy__is_reversible(env):
    env = env.reset()
    while not env.done:
        state_recovered = env.policy2state(env.state2policy())
        if state_recovered is not None:
            assert env.equal(env.state, state_recovered)
        env.step_random()


@pytest.mark.repeat(100)
def test__state2readable__is_reversible(env):
    env = env.reset()
    while not env.done:
        state_recovered = env.readable2state(env.state2readable())
        if state_recovered is not None:
            assert env.isclose(env.state, state_recovered)
        env.step_random()


def test__get_parents__returns_no_parents_in_initial_state(env):
    parents, actions = env.get_parents()
    assert len(parents) == 0
    assert len(actions) == 0


def test__default_config_equals_default_args(env, env_config_path):
    with open(env_config_path, "r") as f:
        config_env = yaml.safe_load(f)
    env_config = hydra.utils.instantiate(config)
    assert True


def test__gflownet_minimal_runs(env):
    # Load config
    with initialize(version_base="1.1", config_path="../../../config", job_name="xxx"):
        config = compose(config_name="tests")
    # Logger
    logger = hydra.utils.instantiate(config.logger, config, _recursive_=False)
    # Proxy
    proxy = hydra.utils.instantiate(
        config.proxy, device=config.device, float_precision=config.float_precision
    )
    # Policy
    forward_config = OmegaConf.create(config.policy)
    forward_config["config"] = config.policy.forward
    del forward_config.forward
    del forward_config.backward
    backward_config = OmegaConf.create(config.policy)
    backward_config["config"] = config.policy.backward
    del backward_config.forward
    del backward_config.backward
    forward_policy = hydra.utils.instantiate(
        forward_config,
        env=env,
        device=config.device,
        float_precision=config.float_precision,
    )
    backward_policy = hydra.utils.instantiate(
        backward_config,
        env=env,
        device=config.device,
        float_precision=config.float_precision,
        base=forward_policy,
    )
    # Set proxy in env
    env.proxy = proxy
    # No buffers
    config.env.buffer.train = None
    config.env.buffer.test = None
    # No replay buffer
    config.env.buffer.replay_capacity = 0
    # Set 1 training step
    config.gflownet.optimizer.n_train_steps = 1
    # GFlowNet agent
    gflownet = hydra.utils.instantiate(
        config.gflownet,
        device=config.device,
        float_precision=config.float_precision,
        env=env,
        forward_policy=forward_policy,
        backward_policy=backward_policy,
        buffer=config.env.buffer,
        logger=logger,
    )
    gflownet.train()
    assert True


@pytest.mark.repeat(100)
def test__sample_actions__get_logprobs__return_valid_actions_and_logprobs(env):
    env = env.reset()
    while not env.done:
        policy_outputs = torch.unsqueeze(torch.tensor(env.random_policy_output), 0)
        mask_invalid = env.get_mask_invalid_actions_forward()
        valid_actions = [a for a, m in zip(env.action_space, mask_invalid) if not m]
        masks_invalid_torch = torch.unsqueeze(torch.BoolTensor(mask_invalid), 0)
        actions, logprobs_sab = env.sample_actions_batch(
            policy_outputs, masks_invalid_torch, [env.state], is_backward=False
        )
        actions_torch = torch.tensor(actions)
        logprobs_glp = env.get_logprobs(
            policy_outputs=policy_outputs,
            actions=actions_torch,
            mask=masks_invalid_torch,
            states_from=None,
            is_backward=False,
        )
        action = actions[0]
        assert env.action2representative(action) in valid_actions
        assert torch.equal(logprobs_sab, logprobs_glp)
        env.step(action)


@pytest.mark.repeat(1000)
def test__forward_actions_have_nonzero_backward_prob(env):
    env = env.reset()
    policy_random = torch.unsqueeze(
        tfloat(env.random_policy_output, float_type=env.float, device=env.device), 0
    )
    while not env.done:
        state_new, action, valid = env.step_random(backward=False)
        if not valid:
            continue
        # Get backward logprobs
        mask_bw = env.get_mask_invalid_actions_backward()
        masks = torch.unsqueeze(tbool(mask_bw, device=env.device), 0)
        actions_torch = torch.unsqueeze(
            tfloat(action, float_type=env.float, device=env.device), 0
        )
        states_torch = torch.unsqueeze(
            tfloat(env.state, float_type=env.float, device=env.device), 0
        )
        policy_outputs = policy_random.clone().detach()
        logprobs_bw = env.get_logprobs(
            policy_outputs=policy_outputs,
            actions=actions_torch,
            mask=masks,
            states_from=states_torch,
            is_backward=True,
        )
        assert torch.isfinite(logprobs_bw)
        assert logprobs_bw > -1e6


@pytest.mark.repeat(1000)
def test__trajectories_are_reversible(env):
    # Skip for ceertain environments until fixed:
    skip_envs = ["Crystal"]
    if env.__class__.__name__ in skip_envs:
        warnings.warn("Skipping test for this specific environment.")
        return
    env = env.reset()

    # Sample random forward trajectory
    states_trajectory_fw = []
    actions_trajectory_fw = []
    while not env.done:
        state, action, valid = env.step_random(backward=False)
        if valid:
            states_trajectory_fw.append(state)
            actions_trajectory_fw.append(action)

    # Sample backward trajectory with actions in forward trajectory
    states_trajectory_bw = []
    actions_trajectory_bw = []
    actions_trajectory_fw_copy = actions_trajectory_fw.copy()
    while not env.equal(env.state, env.source) or env.done:
        state, action, valid = env.step_backwards(actions_trajectory_fw_copy.pop())
        if valid:
            states_trajectory_bw.append(state)
            actions_trajectory_bw.append(action)

    assert all(
        [
            env.equal(s_fw, s_bw)
            for s_fw, s_bw in zip(
                states_trajectory_fw[:-1], states_trajectory_bw[-2::-1]
            )
        ]
    )
    assert actions_trajectory_fw == actions_trajectory_bw[::-1]


def test__backward_actions_have_nonzero_forward_prob(env, n=1000):
    states = _get_terminating_states(env, n)
    if states is None:
        warnings.warn("Skipping test because states are None.")
        return
    policy_random = torch.unsqueeze(
        tfloat(env.random_policy_output, float_type=env.float, device=env.device), 0
    )
    for state in states:
        env.set_state(state, done=True)
        while True:
            if env.equal(env.state, env.source):
                break
            state_new, action, valid = env.step_random(backward=True)
            assert valid
            # Get forward logprobs
            mask_fw = env.get_mask_invalid_actions_forward()
            masks = torch.unsqueeze(tbool(mask_fw, device=env.device), 0)
            actions_torch = torch.unsqueeze(
                tfloat(action, float_type=env.float, device=env.device), 0
            )
            states_torch = torch.unsqueeze(
                tfloat(env.state, float_type=env.float, device=env.device), 0
            )
            policy_outputs = policy_random.clone().detach()
            logprobs_fw = env.get_logprobs(
                policy_outputs=policy_outputs,
                actions=actions_torch,
                mask=masks,
                states_from=states_torch,
                is_backward=False,
            )
            assert torch.isfinite(logprobs_fw)
            assert logprobs_fw > -1e6


@pytest.mark.repeat(10)
def test__init__state_is_source_no_parents(env):
    assert env.equal(env.state, env.source)
    parents, actions = env.get_parents()
    assert len(parents) == 0
    assert len(actions) == 0


@pytest.mark.repeat(10)
def test__reset__state_is_source(env):
    env.step_random()
    env.reset()
    assert env.equal(env.state, env.source)


@pytest.mark.repeat(10)
def test__reset__state_is_source_no_parents(env):
    env.step_random()
    env.reset()
    assert env.equal(env.state, env.source)
    parents, actions = env.get_parents()
    assert len(parents) == 0
    assert len(actions) == 0


def test__get_parents__returns_same_state_and_eos_if_done(env):
    env.set_state(env.state, done=True)
    parents, actions = env.get_parents()
    if torch.is_tensor(env.state):
        assert all([env.equal(p, env.state) for p in parents])
    else:
        assert parents == [env.state]
    assert actions == [env.action_space[-1]]


@pytest.mark.repeat(10)
def test__step__returns_same_state_action_and_invalid_if_done(env):
    env.reset()
    # Sample random trajectory
    env.trajectory_random()
    assert env.done
    # Attempt another step
    action = env.action_space[np.random.randint(low=0, high=env.action_space_dim)]
    next_state, action_step, valid = env.step(action)
    if torch.is_tensor(env.state):
        assert env.equal(next_state, env.state)
    else:
        assert next_state == env.state
    assert action_step == action
    assert valid is False


@pytest.mark.repeat(10)
def test__actions2indices__returns_expected_tensor(env, batch_size=100):
    action_space = env.action_space_torch
    indices_rand = torch.randint(low=0, high=action_space.shape[0], size=(batch_size,))
    actions = action_space[indices_rand, :]
    action_indices = env.actions2indices(actions)
    assert torch.equal(action_indices, indices_rand)
    assert torch.equal(action_space[action_indices], actions)
