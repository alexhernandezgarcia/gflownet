import warnings

import hydra
import numpy as np
import pytest
import torch
import yaml
from hydra import compose, initialize

from gflownet.utils.common import copy


def test__all_env_common(env):
    test__init__state_is_source_no_parents(env)
    test__reset__state_is_source_no_parents(env)
    test__step__returns_same_state_action_and_invalid_if_done(env)
    test__sample_actions__get_logprobs__return_valid_actions_and_logprobs(env)
    test__step_random__does_not_sample_invalid_actions(env)
    test__get_parents_step_get_mask__are_compatible(env)
    test__sample_backwards_reaches_source(env)
    test__state2readable__is_reversible(env)
    test__get_parents__returns_same_state_and_eos_if_done(env)
    test__actions2indices__returns_expected_tensor(env)
    test__gflownet_minimal_runs(env)


def test__continuous_env_common(env):
    test__reset__state_is_source(env)
    test__get_parents__returns_no_parents_in_initial_state(env)
    #     test__gflownet_minimal_runs(env)
    #     test__sample_actions__get_logprobs__return_valid_actions_and_logprobs(env)
    test__get_parents__returns_same_state_and_eos_if_done(env)
    test__step__returns_same_state_action_and_invalid_if_done(env)
    test__actions2indices__returns_expected_tensor(env)


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


@pytest.mark.repeat(100)
def test__sample_backwards_reaches_source(env, n=100):
    if hasattr(env, "get_all_terminating_states"):
        x = env.get_all_terminating_states()
    elif hasattr(env, "get_uniform_terminating_states"):
        x = env.get_uniform_terminating_states(n, 0)
    else:
        print(
            """
        Environment does not have neither get_all_terminating_states() nor
        get_uniform_terminating_states(). Backward sampling will not be tested.
        """
        )
        return
    for state in x:
        env.set_state(state, done=True)
        n_actions = 0
        while True:
            if torch.is_tensor(env.state):
                if torch.equal(env.state, env.source):
                    assert True
                    break
            else:
                if env.state == env.source:
                    assert True
                    break
            parents, parents_a = env.get_parents()
            assert len(parents) > 0
            # Sample random parent
            parent = parents[np.random.permutation(len(parents))[0]]
            env.set_state(parent)
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
        actions, logprobs_sa = env.sample_actions(
            policy_outputs=policy_outputs, mask_invalid_actions=masks_invalid_torch
        )
        actions_torch = torch.tensor(actions)
        logprobs_glp = env.get_logprobs(
            policy_outputs=policy_outputs,
            is_forward=True,
            actions=actions_torch,
            states_target=None,
            mask_invalid_actions=masks_invalid_torch,
        )
        action = actions[0]
        assert env.action2representative(action) in valid_actions
        assert torch.equal(logprobs_sa, logprobs_glp)
        env.step(action)


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
