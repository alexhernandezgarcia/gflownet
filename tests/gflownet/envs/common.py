import pytest
import hydra
import numpy as np
import yaml
from hydra import compose, initialize


@pytest.mark.repeat(100)
def test__get_parents_step_get_mask__are_compatible(env):
    env = env.reset()
    n_actions = 0
    while not env.done:
        state = env.state
        mask_invalid = env.get_mask_invalid_actions_forward()
        valid_actions = [a for a, m in zip(env.action_space, mask_invalid) if not m]
        # Sample random action
        action = tuple(np.random.permutation(valid_actions)[0])
        next_state, action, valid = env.step(action)
        if valid is False:
            continue
        n_actions += 1
        assert n_actions <= env.get_max_traj_len()
        assert env.n_actions == n_actions
        parents, parents_a = env.get_parents()
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
        x = env.get_uniform_terminating_states(n)
    else:
        print(
            """
        Environment does not have neither get_all_terminating_states() nor
        get_uniform_terminating_states(). Backward sampling will not be tested.
        """
        )
        assert True
    for state in x:
        env.set_state(state, done=True)
        n_actions = 0
        while True:
            if env.state == env.source:
                assert True
                break
            parents, parents_a = env.get_parents()
            assert len(parents) > 0
            # Sample random parent
            parent = parents[np.random.permutation(len(parents))[0]]
            env.set_state(parent)
            n_actions += 1
            assert n_actions <= env.get_max_traj_len()


@pytest.mark.repeat(100)
def test__state_conversions_are_reversible(env):
    env = env.reset()
    while not env.done:
        state = env.state
        assert state == env.policy2state(env.state2policy(state))
        assert state == env.readable2state(env.state2readable(state))
        mask_invalid = env.get_mask_invalid_actions_forward()
        valid_actions = [a for a, m in zip(env.action_space, mask_invalid) if not m]
        action = tuple(np.random.permutation(valid_actions)[0])
        env.step(action)


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
    # No buffer
    config.env.buffer.train = None
    config.env.buffer.test = None
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
