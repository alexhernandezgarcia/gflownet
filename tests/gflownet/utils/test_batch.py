import numpy as np
import pytest
import torch

from gflownet.envs.ctorus import ContinuousTorus
from gflownet.envs.grid import Grid
from gflownet.envs.tetris import Tetris
from gflownet.proxy.box.corners import Corners
from gflownet.proxy.tetris import Tetris as TetrisScore
from gflownet.utils.batch import Batch
from gflownet.utils.common import (
    concat_items,
    copy,
    set_device,
    set_float_precision,
    tbool,
    tfloat,
    tint,
    tlong,
)

# Sets the number of repetitions for the tests. Please increase to ~10 after
# introducing changes to the Batch class and decrease again to 1 when passed.
N_REPETITIONS = 2
# Sets the batch size for the tests. Please increase to ~10 after introducing changes
# to the Batch class and decrease again to 5 when passed.
BATCH_SIZE = 5


@pytest.fixture
def batch():
    return Batch()


@pytest.fixture
def grid2d():
    """
    During development, consider testing with a larger environment too, for example:

    Grid(n_dim=5, length=10, cell_min=-1.0, cell_max=1.0)
    """
    return Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0)


@pytest.fixture
def tetris6x4():
    """
    During development, consider testing with a larger environment too, for example:

    Tetris(width=10, height=20, device="cpu")
    """
    return Tetris(width=6, height=4, device="cpu")


@pytest.fixture
def ctorus2d5l():
    """
    During development, consider testing with a larger environment too, for example:

    ContinuousTorus(n_dim=5, length_traj=10, n_comp=2)
    """
    return ContinuousTorus(n_dim=2, length_traj=10, n_comp=2)


@pytest.fixture()
def corners():
    return Corners(device="cpu", float_precision=32, mu=0.75, sigma=0.05)


@pytest.fixture()
def tetris_score():
    return TetrisScore(device="cpu", float_precision=32, normalize=False)


@pytest.fixture()
def tetris_score_norm():
    return TetrisScore(device="cpu", float_precision=32, normalize=True)


# @pytest.mark.skip(reason="skip while developping other tests")
def test__len__returnszero_at_init(batch):
    assert len(batch) == 0


@pytest.mark.repeat(N_REPETITIONS)
@pytest.mark.parametrize("env", ["grid2d", "tetris6x4", "ctorus2d5l"])
# @pytest.mark.skip(reason="skip while developping other tests")
def test__add_to_batch__single_env_adds_expected(env, batch, request):
    env = request.getfixturevalue(env)
    env = env.reset()
    batch.set_env(env)
    while not env.done:
        # Sample random action
        state, action, valid = env.step_random()
        # Add to batch
        batch.add_to_batch([env], [action], [valid])
        if valid is False:
            continue
        # Checks
        assert batch.traj_indices[-1] == env.id
        if torch.is_tensor(state):
            assert torch.equal(batch.states[-1], state)
        else:
            assert batch.states[-1] == state
        assert batch.actions[-1] == action
        assert batch.done[-1] == env.done
        assert batch.state_indices[-1] == env.n_actions


@pytest.mark.repeat(N_REPETITIONS)
@pytest.mark.parametrize("env", ["grid2d", "tetris6x4", "ctorus2d5l"])
# @pytest.mark.skip(reason="skip while developping other tests")
def test__get_states__single_env_returns_expected(env, batch, request):
    env = request.getfixturevalue(env)
    env = env.reset()
    batch.set_env(env)
    states = []
    while not env.done:
        # Sample random action
        state, action, valid = env.step_random()
        # Add to batch
        batch.add_to_batch([env], [action], [valid])
        if valid:
            states.append(copy(state))
    states_batch = batch.get_states()
    states_policy_batch = batch.get_states(policy=True)
    if torch.is_tensor(states[0]):
        assert torch.equal(torch.stack(states_batch), torch.stack(states))
    else:
        assert states_batch == states
    assert torch.equal(states_policy_batch, env.states2policy(states))


@pytest.mark.repeat(N_REPETITIONS)
@pytest.mark.parametrize("env", ["grid2d", "tetris6x4", "ctorus2d5l"])
# @pytest.mark.skip(reason="skip while developping other tests")
def test__get_parents__single_env_returns_expected(env, batch, request):
    env = request.getfixturevalue(env)
    env = env.reset()
    batch.set_env(env)
    parents = []
    while not env.done:
        parent = copy(env.state)
        # Sample random action
        _, action, valid = env.step_random()
        # Add to batch
        batch.add_to_batch([env], [action], [valid])
        if valid:
            parents.append(parent)
    parents_batch = batch.get_parents()
    parents_policy_batch = batch.get_parents(policy=True)
    if torch.is_tensor(parents[0]):
        assert torch.equal(torch.stack(parents_batch), torch.stack(parents))
    else:
        assert parents_batch == parents
    assert torch.equal(parents_policy_batch, env.states2policy(parents))


@pytest.mark.repeat(N_REPETITIONS)
@pytest.mark.parametrize("env", ["grid2d", "tetris6x4"])
# @pytest.mark.skip(reason="skip while developping other tests")
def test__get_parents_all__single_env_returns_expected(env, batch, request):
    env = request.getfixturevalue(env)
    env = env.reset()
    batch.set_env(env)
    parents_all = []
    parents_all_a = []
    while not env.done:
        # Sample random action
        _, action, valid = env.step_random()
        # Add to batch
        batch.add_to_batch([env], [action], [valid])
        if valid:
            parents, parents_a = env.get_parents()
            parents_all.extend(parents)
            parents_all_a.extend(parents_a)
    parents_all_batch, parents_all_a_batch, _ = batch.get_parents_all()
    parents_all_policy_batch, _, _ = batch.get_parents_all(policy=True)
    if torch.is_tensor(parents_all[0]):
        assert torch.equal(torch.stack(parents_all_batch), torch.stack(parents_all))
    else:
        assert parents_all_batch == parents_all
    assert torch.equal(
        parents_all_a_batch,
        tfloat(
            parents_all_a,
            device=batch.device,
            float_type=batch.float,
        ),
    )
    assert torch.equal(parents_all_policy_batch, env.states2policy(parents_all))


@pytest.mark.repeat(N_REPETITIONS)
@pytest.mark.parametrize("env", ["grid2d", "tetris6x4", "ctorus2d5l"])
# @pytest.mark.skip(reason="skip while developping other tests")
def test__get_masks_forward__single_env_returns_expected(env, batch, request):
    env = request.getfixturevalue(env)
    env = env.reset()
    batch.set_env(env)
    masks_forward = []
    while not env.done:
        parent = env.state
        # Sample random action
        _, action, valid = env.step_random()
        # Add to batch
        batch.add_to_batch([env], [action], [valid])
        if valid:
            masks_forward.append(env.get_mask_invalid_actions_forward())
    masks_forward_batch = batch.get_masks_forward()
    assert torch.equal(masks_forward_batch, tbool(masks_forward, device=batch.device))


@pytest.mark.repeat(N_REPETITIONS)
@pytest.mark.parametrize("env", ["grid2d", "tetris6x4", "ctorus2d5l"])
# @pytest.mark.skip(reason="skip while developping other tests")
def test__get_masks_backward__single_env_returns_expected(env, batch, request):
    env = request.getfixturevalue(env)
    env = env.reset()
    batch.set_env(env)
    masks_backward = []
    while not env.done:
        parent = env.state
        # Sample random action
        _, action, valid = env.step_random()
        # Add to batch
        batch.add_to_batch([env], [action], [valid])
        if valid:
            masks_backward.append(env.get_mask_invalid_actions_backward())
    masks_backward_batch = batch.get_masks_backward()
    assert torch.equal(masks_backward_batch, tbool(masks_backward, device=batch.device))


@pytest.mark.repeat(N_REPETITIONS)
@pytest.mark.parametrize(
    "env, proxy",
    [("grid2d", "corners"), ("tetris6x4", "tetris_score"), ("ctorus2d5l", "corners")],
)
def test__get_rewards__single_env_returns_expected(env, proxy, batch, request):
    env = request.getfixturevalue(env)
    proxy = request.getfixturevalue(proxy)
    proxy.setup(env)
    env = env.reset()
    batch.set_env(env)
    batch.set_proxy(proxy)

    rewards = []
    while not env.done:
        parent = env.state
        # Sample random action
        _, action, valid = env.step_random()
        # Add to batch
        batch.add_to_batch([env], [action], [valid])
        if valid:
            if env.done:
                rewards.append(proxy.rewards(env.state2proxy())[0])
            else:
                rewards.append(
                    tfloat(
                        proxy.get_min_reward(),
                        float_type=batch.float,
                        device=batch.device,
                    )
                )
    rewards_batch = batch.get_rewards()
    rewards = torch.stack(rewards)
    assert torch.equal(
        rewards_batch,
        tfloat(rewards, device=batch.device, float_type=batch.float),
    ), (rewards, rewards_batch)


@pytest.mark.repeat(N_REPETITIONS)
@pytest.mark.parametrize(
    "env, proxy",
    [("grid2d", "corners"), ("tetris6x4", "tetris_score"), ("ctorus2d5l", "corners")],
)
def test__get_logrewards__single_env_returns_expected(env, proxy, batch, request):
    env = request.getfixturevalue(env)
    proxy = request.getfixturevalue(proxy)
    proxy.setup(env)
    env = env.reset()
    batch.set_env(env)
    batch.set_proxy(proxy)

    logrewards = []
    while not env.done:
        parent = env.state
        # Sample random action
        _, action, valid = env.step_random()
        # Add to batch
        batch.add_to_batch([env], [action], [valid])
        if valid:
            if env.done:
                logrewards.append(proxy.rewards(env.state2proxy(), log=True)[0])
            else:
                logrewards.append(
                    tfloat(
                        proxy.get_min_reward(log=True),
                        float_type=batch.float,
                        device=batch.device,
                    )
                )
    logrewards_batch = batch.get_rewards(log=True)
    logrewards = torch.stack(logrewards)
    assert torch.equal(
        logrewards_batch,
        tfloat(logrewards, device=batch.device, float_type=batch.float),
    ), (logrewards, logrewards_batch)


@pytest.mark.repeat(N_REPETITIONS)
@pytest.mark.parametrize(
    "env, proxy",
    [("grid2d", "corners"), ("tetris6x4", "tetris_score"), ("ctorus2d5l", "corners")],
)
# @pytest.mark.skip(reason="skip while developping other tests")
def test__forward_sampling_multiple_envs_all_as_expected(env, proxy, batch, request):
    batch_size = BATCH_SIZE
    env_ref = request.getfixturevalue(env)
    proxy = request.getfixturevalue(proxy)
    proxy.setup(env_ref)
    batch.set_env(env_ref)
    batch.set_proxy(proxy)

    # Make list of envs
    envs = []
    for idx in range(batch_size):
        env_aux = env_ref.copy().reset(idx)
        envs.append(env_aux)

    # Initialize empty lists for checks
    states = []
    actions = []
    done = []
    masks_forward = []
    masks_parents_forward = []
    masks_backward = []
    parents = []
    parents_all = []
    parents_all_a = []
    rewards = []
    proxy_values = []
    traj_indices = []
    state_indices = []
    states_term_sorted = [None for _ in range(batch_size)]

    # Iterate until envs is empty
    while envs:
        actions_iter = []
        valids_iter = []
        # Make step env by env (different to GFN Agent) to have full control
        for env in envs:
            parent = copy(env.state)
            # Sample random action
            state, action, valid = env.step_random()
            if valid:
                # Add to iter lists
                actions_iter.append(action)
                valids_iter.append(valid)
                # Add to checking lists
                states.append(copy(env.state))
                actions.append(action)
                done.append(env.done)
                masks_forward.append(env.get_mask_invalid_actions_forward())
                masks_parents_forward.append(
                    env.get_mask_invalid_actions_forward(parent, done=False)
                )
                masks_backward.append(env.get_mask_invalid_actions_backward())
                parents.append(parent)
                if not env.continuous:
                    env_parents, env_parents_a = env.get_parents()
                    parents_all.extend(env_parents)
                    parents_all_a.extend(env_parents_a)
                if env.done:
                    reward, proxy_value = proxy.rewards(
                        env.state2proxy(), return_proxy=True
                    )
                    rewards.append(reward[0])
                    proxy_values.append(proxy_value[0])
                else:
                    rewards.append(
                        tfloat(
                            proxy.get_min_reward(),
                            float_type=batch.float,
                            device=batch.device,
                        )
                    )
                    proxy_values.append(
                        tfloat(torch.inf, float_type=batch.float, device=batch.device)
                    )
                traj_indices.append(env.id)
                state_indices.append(env.n_actions)
                if env.done:
                    states_term_sorted[env.id] = env.state
        # Add all envs, actions and valids to batch
        batch.add_to_batch(envs, actions_iter, valids_iter)
        # Remove done envs
        envs = [env for env in envs if not env.done]

    # Check trajectory indices
    traj_indices_batch = batch.get_trajectory_indices()
    assert torch.equal(traj_indices_batch, tlong(traj_indices, device=batch.device))
    # Check state indices
    state_indices_batch = batch.get_state_indices()
    assert torch.equal(state_indices_batch, tlong(state_indices, device=batch.device))
    # Check states
    states_batch = batch.get_states()
    states_policy_batch = batch.get_states(policy=True)
    if torch.is_tensor(states[0]):
        assert torch.equal(torch.stack(states_batch), torch.stack(states))
    else:
        assert states_batch == states
    assert torch.equal(states_policy_batch, env.states2policy(states))
    # Check actions
    actions_batch = batch.get_actions()
    assert torch.equal(
        actions_batch, tfloat(actions, float_type=batch.float, device=batch.device)
    )
    # Check done
    done_batch = batch.get_done()
    assert torch.equal(done_batch, tbool(done, device=batch.device))
    # Check masks forward
    masks_forward_batch = batch.get_masks_forward()
    assert torch.equal(masks_forward_batch, tbool(masks_forward, device=batch.device))
    # Check masks parents forward
    masks_parents_forward_batch = batch.get_masks_forward(of_parents=True)
    assert torch.equal(
        masks_parents_forward_batch, tbool(masks_parents_forward, device=batch.device)
    )
    # Check masks backward
    masks_backward_batch = batch.get_masks_backward()
    assert torch.equal(masks_backward_batch, tbool(masks_backward, device=batch.device))
    # Check parents
    parents_batch = batch.get_parents()
    parents_policy_batch = batch.get_parents(policy=True)
    if torch.is_tensor(parents[0]):
        assert torch.equal(torch.stack(parents_batch), torch.stack(parents))
    else:
        assert parents_batch == parents
    assert torch.equal(parents_policy_batch, env.states2policy(parents))
    # Check parents_all
    if not env.continuous:
        parents_all_batch, parents_all_a_batch, _ = batch.get_parents_all()
        parents_all_policy_batch, _, _ = batch.get_parents_all(policy=True)
        if torch.is_tensor(parents_all[0]):
            assert torch.equal(torch.stack(parents_all_batch), torch.stack(parents_all))
        else:
            assert parents_all_batch == parents_all
        assert torch.equal(
            parents_all_a_batch,
            tfloat(
                parents_all_a,
                device=batch.device,
                float_type=batch.float,
            ),
        )
        assert torch.equal(parents_all_policy_batch, env.states2policy(parents_all))
    # Check rewards
    rewards_batch = batch.get_rewards()
    rewards = torch.stack(rewards)
    assert torch.all(
        torch.isclose(
            rewards_batch,
            tfloat(rewards, device=batch.device, float_type=batch.float),
        )
    ), (rewards, rewards_batch)
    # Check proxy values
    proxy_values_batch = batch.get_proxy_values()
    proxy_values = torch.stack(proxy_values)
    assert torch.all(
        torch.isclose(
            proxy_values_batch,
            tfloat(proxy_values, device=batch.device, float_type=batch.float),
        )
    ), (proxy_values, proxy_values_batch)
    # Check terminating states (sorted by trajectory)
    states_term_batch = batch.get_terminating_states(sort_by="traj")
    states_term_policy_batch = batch.get_terminating_states(sort_by="traj", policy=True)
    if torch.is_tensor(states_term_sorted[0]):
        assert torch.equal(
            torch.stack(states_term_batch), torch.stack(states_term_sorted)
        )
    else:
        assert states_term_batch == states_term_sorted
    assert torch.equal(states_term_policy_batch, env.states2policy(states_term_sorted))


@pytest.mark.repeat(N_REPETITIONS)
@pytest.mark.parametrize(
    "env, proxy",
    [("grid2d", "corners"), ("tetris6x4", "tetris_score")],
)
# @pytest.mark.skip(reason="skip while developping other tests")
def test__backward_sampling_multiple_envs_all_as_expected(env, proxy, batch, request):
    batch_size = BATCH_SIZE
    env_ref = request.getfixturevalue(env)
    proxy = request.getfixturevalue(proxy)
    proxy.setup(env_ref)
    batch.set_env(env_ref)
    batch.set_proxy(proxy)

    # Sample terminating states and build list of envs
    x_batch = env_ref.get_random_terminating_states(n_states=batch_size)
    envs = []
    for idx, x in enumerate(x_batch):
        env_aux = env_ref.copy().reset(idx)
        env_aux = env_aux.set_state(state=x, done=True)
        env_aux.n_actions = env_aux.get_max_traj_length()
        envs.append(env_aux)

    # Initialize empty lists for checks
    states = []
    actions = []
    dones = []
    masks_forward = []
    masks_parents_forward = []
    masks_backward = []
    parents = []
    parents_all = []
    parents_all_a = []
    rewards = []
    proxy_values = []
    traj_indices = []
    state_indices = []
    states_term_sorted = [copy(x) for x in x_batch]

    # Iterate until envs is empty
    while envs:
        actions_iter = []
        valids_iter = []
        # Make step env by env (different to GFN Agent) to have full control
        for env in envs:
            # Compute variables before performing action
            state = copy(env.state)
            done = env.done
            mask_forward = env.get_mask_invalid_actions_forward()
            mask_backward = env.get_mask_invalid_actions_backward()
            if not env.continuous:
                env_parents, env_parents_a = env.get_parents()
            if env.done:
                reward, proxy_value = proxy.rewards(
                    env.state2proxy(), return_proxy=True
                )
                reward = reward[0]
                proxy_value = proxy_value[0]
            else:
                reward = tfloat(
                    proxy.get_min_reward(), float_type=batch.float, device=batch.device
                )
                proxy_value = tfloat(
                    torch.inf, float_type=batch.float, device=batch.device
                )
            if env.done:
                states_term_sorted[env.id] = env.state
            # Sample random action
            parent, action, valid = env.step_random(backward=True)
            if valid:
                # Add to iter lists
                actions_iter.append(action)
                valids_iter.append(valid)
                # Add to checking lists
                states.append(state)
                actions.append(action)
                dones.append(done)
                masks_forward.append(mask_forward)
                masks_parents_forward.append(env.get_mask_invalid_actions_forward())
                masks_backward.append(mask_backward)
                parents.append(parent)
                if not env.continuous:
                    parents_all.extend(env_parents)
                    parents_all_a.extend(env_parents_a)
                rewards.append(reward)
                proxy_values.append(proxy_value)
                traj_indices.append(env.id)
                state_indices.append(env.n_actions)
        # Add all envs, actions and valids to batch
        batch.add_to_batch(envs, actions_iter, valids_iter, backward=True)
        # Remove finished trajectories (state reached source)
        envs = [env for env in envs if not env.equal(env.state, env.source)]

    # Check trajectory indices
    traj_indices_batch = batch.get_trajectory_indices()
    assert torch.equal(traj_indices_batch, tlong(traj_indices, device=batch.device))
    # Check state indices
    state_indices_batch = batch.get_state_indices()
    assert torch.equal(state_indices_batch, tlong(state_indices, device=batch.device))
    # Check states
    states_batch = batch.get_states()
    states_policy_batch = batch.get_states(policy=True)
    if torch.is_tensor(states[0]):
        assert torch.equal(torch.stack(states_batch), torch.stack(states))
    else:
        assert states_batch == states
    assert torch.equal(states_policy_batch, env.states2policy(states))
    # Check actions
    actions_batch = batch.get_actions()
    assert torch.equal(
        actions_batch, tfloat(actions, float_type=batch.float, device=batch.device)
    )
    # Check done
    done_batch = batch.get_done()
    assert torch.equal(done_batch, tbool(dones, device=batch.device))
    # Check masks forward
    masks_forward_batch = batch.get_masks_forward()
    assert torch.equal(masks_forward_batch, tbool(masks_forward, device=batch.device))
    # Check masks parents forward
    masks_parents_forward_batch = batch.get_masks_forward(of_parents=True)
    assert torch.equal(
        masks_parents_forward_batch, tbool(masks_parents_forward, device=batch.device)
    )
    # Check masks backward
    masks_backward_batch = batch.get_masks_backward()
    assert torch.equal(masks_backward_batch, tbool(masks_backward, device=batch.device))
    # Check parents
    parents_batch = batch.get_parents()
    parents_policy_batch = batch.get_parents(policy=True)
    if torch.is_tensor(parents[0]):
        assert torch.equal(torch.stack(parents_batch), torch.stack(parents))
    else:
        assert parents_batch == parents
    assert torch.equal(parents_policy_batch, env.states2policy(parents))
    # Check parents_all
    if not env.continuous:
        parents_all_batch, parents_all_a_batch, _ = batch.get_parents_all()
        parents_all_policy_batch, _, _ = batch.get_parents_all(policy=True)
        if torch.is_tensor(parents_all[0]):
            assert torch.equal(torch.stack(parents_all_batch), torch.stack(parents_all))
        else:
            assert parents_all_batch == parents_all
        assert torch.equal(
            parents_all_a_batch,
            tfloat(
                parents_all_a,
                device=batch.device,
                float_type=batch.float,
            ),
        )
        assert torch.equal(parents_all_policy_batch, env.states2policy(parents_all))
    # Check rewards
    rewards_batch = batch.get_rewards()
    rewards = torch.stack(rewards)
    assert torch.all(
        torch.isclose(
            rewards_batch,
            tfloat(rewards, device=batch.device, float_type=batch.float),
        )
    ), (rewards, rewards_batch)
    # Check proxy values
    proxy_values_batch = batch.get_proxy_values()
    proxy_values = torch.stack(proxy_values)
    assert torch.all(
        torch.isclose(
            proxy_values_batch,
            tfloat(proxy_values, device=batch.device, float_type=batch.float),
        )
    ), (proxy_values, proxy_values_batch)
    # Check terminating states (sorted by trajectory)
    states_term_batch = batch.get_terminating_states(sort_by="traj")
    states_term_policy_batch = batch.get_terminating_states(sort_by="traj", policy=True)
    if torch.is_tensor(states_term_sorted[0]):
        assert torch.equal(
            torch.stack(states_term_batch), torch.stack(states_term_sorted)
        )
    else:
        assert states_term_batch == states_term_sorted
    assert torch.equal(states_term_policy_batch, env.states2policy(states_term_sorted))


@pytest.mark.repeat(N_REPETITIONS)
@pytest.mark.parametrize(
    "env, proxy",
    [("grid2d", "corners"), ("tetris6x4", "tetris_score")],
)
# @pytest.mark.skip(reason="skip while developping other tests")
def test__mixed_sampling_multiple_envs_all_as_expected(env, proxy, batch, request):
    # Initialize fixtures and batch
    env_ref = request.getfixturevalue(env)
    proxy = request.getfixturevalue(proxy)
    proxy.setup(env_ref)
    batch.set_env(env_ref)
    batch.set_proxy(proxy)

    # Initialize empty lists for checks
    states = []
    actions = []
    dones = []
    masks_forward = []
    masks_parents_forward = []
    masks_backward = []
    parents = []
    parents_all = []
    parents_all_a = []
    rewards = []
    proxy_values = []
    traj_indices = []
    state_indices = []
    states_term_sorted = []

    ### FORWARD ###

    # Make list of envs
    batch_size_forward = BATCH_SIZE
    envs = []
    for idx in range(batch_size_forward):
        env_aux = env_ref.copy().reset(idx)
        envs.append(env_aux)

    states_term_sorted.extend([None for _ in range(batch_size_forward)])

    # Iterate until envs is empty
    while envs:
        actions_iter = []
        valids_iter = []
        # Make step env by env (different to GFN Agent) to have full control
        for env in envs:
            parent = copy(env.state)
            # Sample random action
            state, action, valid = env.step_random()
            if valid:
                # Add to iter lists
                actions_iter.append(action)
                valids_iter.append(valid)
                # Add to checking lists
                states.append(copy(env.state))
                actions.append(action)
                dones.append(env.done)
                masks_forward.append(env.get_mask_invalid_actions_forward())
                masks_parents_forward.append(
                    env.get_mask_invalid_actions_forward(parent, done=False)
                )
                masks_backward.append(env.get_mask_invalid_actions_backward())
                parents.append(parent)
                if not env.continuous:
                    env_parents, env_parents_a = env.get_parents()
                    parents_all.extend(env_parents)
                    parents_all_a.extend(env_parents_a)
                if env.done:
                    reward, proxy_value = proxy.rewards(
                        env.state2proxy(), return_proxy=True
                    )
                    rewards.append(reward[0])
                    proxy_values.append(proxy_value[0])
                else:
                    rewards.append(
                        tfloat(
                            proxy.get_min_reward(),
                            float_type=batch.float,
                            device=batch.device,
                        )
                    )
                    proxy_values.append(
                        tfloat(torch.inf, float_type=batch.float, device=batch.device)
                    )
                traj_indices.append(env.id)
                state_indices.append(env.n_actions)
                if env.done:
                    states_term_sorted[env.id] = env.state
        # Add all envs, actions and valids to batch
        batch.add_to_batch(envs, actions_iter, valids_iter)
        # Remove done envs
        envs = [env for env in envs if not env.done]

    ### BACKWARD ###

    # Sample terminating states and build list of envs
    batch_size_backward = BATCH_SIZE
    x_batch = env_ref.get_random_terminating_states(n_states=batch_size_backward)
    envs = []
    for idx, x in enumerate(x_batch):
        env_aux = env_ref.copy().reset(idx + batch_size_forward)
        env_aux = env_aux.set_state(state=x, done=True)
        env_aux.n_actions = env_aux.get_max_traj_length()
        envs.append(env_aux)

    states_term_sorted.extend([copy(x) for x in x_batch])

    # Iterate until envs is empty
    while envs:
        actions_iter = []
        valids_iter = []
        # Make step env by env (different to GFN Agent) to have full control
        for env in envs:
            # Compute variables before performing action
            state = copy(env.state)
            done = env.done
            mask_forward = env.get_mask_invalid_actions_forward()
            mask_backward = env.get_mask_invalid_actions_backward()
            if not env.continuous:
                env_parents, env_parents_a = env.get_parents()
            if env.done:
                reward, proxy_value = proxy.rewards(
                    env.state2proxy(), return_proxy=True
                )
                reward = reward[0]
                proxy_value = proxy_value[0]
            else:
                reward = tfloat(
                    proxy.get_min_reward(), float_type=batch.float, device=batch.device
                )
                proxy_value = tfloat(
                    torch.inf, float_type=batch.float, device=batch.device
                )
            if env.done:
                states_term_sorted[env.id] = env.state
            # Sample random action
            parent, action, valid = env.step_random(backward=True)
            if valid:
                # Add to iter lists
                actions_iter.append(action)
                valids_iter.append(valid)
                # Add to checking lists
                states.append(state)
                actions.append(action)
                dones.append(done)
                masks_forward.append(mask_forward)
                masks_parents_forward.append(env.get_mask_invalid_actions_forward())
                masks_backward.append(mask_backward)
                parents.append(parent)
                if not env.continuous:
                    parents_all.extend(env_parents)
                    parents_all_a.extend(env_parents_a)
                rewards.append(reward)
                proxy_values.append(proxy_value)
                traj_indices.append(env.id)
                state_indices.append(env.n_actions)
        # Add all envs, actions and valids to batch
        batch.add_to_batch(envs, actions_iter, valids_iter, backward=True)
        # Remove finished trajectories (state reached source)
        envs = [env for env in envs if not env.equal(env.state, env.source)]

    ### CHECKS ###

    # Check trajectory indices
    traj_indices_batch = batch.get_trajectory_indices()
    assert torch.equal(traj_indices_batch, tlong(traj_indices, device=batch.device))
    # Check state indices
    state_indices_batch = batch.get_state_indices()
    assert torch.equal(state_indices_batch, tlong(state_indices, device=batch.device))
    # Check states
    states_batch = batch.get_states()
    states_policy_batch = batch.get_states(policy=True)
    if torch.is_tensor(states[0]):
        assert torch.equal(torch.stack(states_batch), torch.stack(states))
    else:
        assert states_batch == states
    assert torch.equal(states_policy_batch, env.states2policy(states))
    # Check actions
    actions_batch = batch.get_actions()
    assert torch.equal(
        actions_batch, tfloat(actions, float_type=batch.float, device=batch.device)
    )
    # Check done
    done_batch = batch.get_done()
    assert torch.equal(done_batch, tbool(dones, device=batch.device))
    # Check masks forward
    masks_forward_batch = batch.get_masks_forward()
    assert torch.equal(masks_forward_batch, tbool(masks_forward, device=batch.device))
    # Check masks parents forward
    masks_parents_forward_batch = batch.get_masks_forward(of_parents=True)
    assert torch.equal(
        masks_parents_forward_batch, tbool(masks_parents_forward, device=batch.device)
    )
    # Check masks backward
    masks_backward_batch = batch.get_masks_backward()
    assert torch.equal(masks_backward_batch, tbool(masks_backward, device=batch.device))
    # Check parents
    parents_batch = batch.get_parents()
    parents_policy_batch = batch.get_parents(policy=True)
    if torch.is_tensor(parents[0]):
        assert torch.equal(torch.stack(parents_batch), torch.stack(parents))
    else:
        assert parents_batch == parents
    assert torch.equal(parents_policy_batch, env.states2policy(parents))
    # Check parents_all
    if not env.continuous:
        parents_all_batch, parents_all_a_batch, _ = batch.get_parents_all()
        parents_all_policy_batch, _, _ = batch.get_parents_all(policy=True)
        if torch.is_tensor(parents_all[0]):
            assert torch.equal(torch.stack(parents_all_batch), torch.stack(parents_all))
        else:
            assert parents_all_batch == parents_all
        assert torch.equal(
            parents_all_a_batch,
            tfloat(
                parents_all_a,
                device=batch.device,
                float_type=batch.float,
            ),
        )
        assert torch.equal(parents_all_policy_batch, env.states2policy(parents_all))
    # Check rewards
    rewards_batch = batch.get_rewards()
    rewards = torch.stack(rewards)
    assert torch.all(
        torch.isclose(
            rewards_batch,
            tfloat(rewards, device=batch.device, float_type=batch.float),
        )
    ), (rewards, rewards_batch)
    # Check proxy values
    proxy_values_batch = batch.get_proxy_values()
    proxy_values = torch.stack(proxy_values)
    assert torch.all(
        torch.isclose(
            proxy_values_batch,
            tfloat(proxy_values, device=batch.device, float_type=batch.float),
        )
    ), (proxy_values, proxy_values_batch)
    # Check terminating states (sorted by trajectory)
    states_term_batch = batch.get_terminating_states(sort_by="traj")
    states_term_policy_batch = batch.get_terminating_states(sort_by="traj", policy=True)
    if torch.is_tensor(states_term_sorted[0]):
        assert torch.equal(
            torch.stack(states_term_batch), torch.stack(states_term_sorted)
        )
    else:
        assert states_term_batch == states_term_sorted
    assert torch.equal(states_term_policy_batch, env.states2policy(states_term_sorted))


@pytest.mark.repeat(N_REPETITIONS)
@pytest.mark.parametrize(
    "env, proxy",
    [("grid2d", "corners"), ("tetris6x4", "tetris_score")],
)
# @pytest.mark.skip(reason="skip while developping other tests")
def test__mixed_sampling_merged_all_as_expected(env, proxy, request):
    # Initialize fixtures and batch
    env_ref = request.getfixturevalue(env)
    proxy = request.getfixturevalue(proxy)
    proxy.setup(env_ref)
    batch_fw = Batch(env=env_ref, proxy=proxy)
    batch_bw = Batch(env=env_ref, proxy=proxy)

    # Initialize empty lists for checks
    states = []
    actions = []
    dones = []
    masks_forward = []
    masks_parents_forward = []
    masks_backward = []
    parents = []
    parents_all = []
    parents_all_a = []
    rewards = []
    proxy_values = []
    traj_indices = []
    state_indices = []
    states_term_sorted = []

    ### FORWARD ###

    # Make list of envs
    batch_size_forward = BATCH_SIZE
    envs = []
    for idx in range(batch_size_forward):
        env_aux = env_ref.copy().reset(idx)
        envs.append(env_aux)

    states_term_sorted.extend([None for _ in range(batch_size_forward)])

    # Iterate until envs is empty
    while envs:
        actions_iter = []
        valids_iter = []
        # Make step env by env (different to GFN Agent) to have full control
        for env in envs:
            parent = copy(env.state)
            # Sample random action
            state, action, valid = env.step_random()
            if valid:
                # Add to iter lists
                actions_iter.append(action)
                valids_iter.append(valid)
                # Add to checking lists
                states.append(copy(env.state))
                actions.append(action)
                dones.append(env.done)
                masks_forward.append(env.get_mask_invalid_actions_forward())
                masks_parents_forward.append(
                    env.get_mask_invalid_actions_forward(parent, done=False)
                )
                masks_backward.append(env.get_mask_invalid_actions_backward())
                parents.append(parent)
                if not env.continuous:
                    env_parents, env_parents_a = env.get_parents()
                    parents_all.extend(env_parents)
                    parents_all_a.extend(env_parents_a)
                if env.done:
                    reward, proxy_value = proxy.rewards(
                        env.state2proxy(), return_proxy=True
                    )
                    rewards.append(reward[0])
                    proxy_values.append(proxy_value[0])
                else:
                    rewards.append(
                        tfloat(
                            proxy.get_min_reward(),
                            float_type=batch_fw.float,
                            device=batch_fw.device,
                        )
                    )
                    proxy_values.append(
                        tfloat(
                            torch.inf, float_type=batch_fw.float, device=batch_fw.device
                        )
                    )
                traj_indices.append(env.id)
                state_indices.append(env.n_actions)
                if env.done:
                    states_term_sorted[env.id] = env.state
        # Add all envs, actions and valids to batch
        batch_fw.add_to_batch(envs, actions_iter, valids_iter)
        # Remove done envs
        envs = [env for env in envs if not env.done]

    ### BACKWARD ###

    # Sample terminating states and build list of envs
    batch_size_backward = BATCH_SIZE
    x_batch = env_ref.get_random_terminating_states(n_states=batch_size_backward)
    envs = []
    for idx, x in enumerate(x_batch):
        env_aux = env_ref.copy().reset(idx)
        env_aux = env_aux.set_state(state=x, done=True)
        env_aux.n_actions = env_aux.get_max_traj_length()
        envs.append(env_aux)

    states_term_sorted.extend([copy(x) for x in x_batch])

    # Iterate until envs is empty
    while envs:
        actions_iter = []
        valids_iter = []
        # Make step env by env (different to GFN Agent) to have full control
        for env in envs:
            # Compute variables before performing action
            state = copy(env.state)
            done = env.done
            mask_forward = env.get_mask_invalid_actions_forward()
            mask_backward = env.get_mask_invalid_actions_backward()
            if not env.continuous:
                env_parents, env_parents_a = env.get_parents()
            if env.done:
                reward, proxy_value = proxy.rewards(
                    env.state2proxy(), return_proxy=True
                )
                reward = reward[0]
                proxy_value = proxy_value[0]
            else:
                reward = tfloat(
                    proxy.get_min_reward(),
                    float_type=batch_bw.float,
                    device=batch_bw.device,
                )
                proxy_value = tfloat(
                    torch.inf, float_type=batch_bw.float, device=batch_bw.device
                )
            if env.done:
                states_term_sorted[env.id + batch_size_forward] = env.state
            # Sample random action
            parent, action, valid = env.step_random(backward=True)
            if valid:
                # Add to iter lists
                actions_iter.append(action)
                valids_iter.append(valid)
                # Add to checking lists
                states.append(state)
                actions.append(action)
                dones.append(done)
                masks_forward.append(mask_forward)
                masks_parents_forward.append(env.get_mask_invalid_actions_forward())
                masks_backward.append(mask_backward)
                parents.append(parent)
                if not env.continuous:
                    parents_all.extend(env_parents)
                    parents_all_a.extend(env_parents_a)
                rewards.append(reward)
                proxy_values.append(proxy_value)
                traj_indices.append(env.id + batch_size_forward)
                state_indices.append(env.n_actions)
        # Add all envs, actions and valids to batch
        batch_bw.add_to_batch(envs, actions_iter, valids_iter, backward=True)
        # Remove finished trajectories (state reached source)
        envs = [env for env in envs if not env.equal(env.state, env.source)]

    ### MERGE BATCHES ###

    batch = Batch(env=env_ref, proxy=proxy)
    batch = batch.merge([batch_fw, batch_bw])

    ### CHECKS ###

    # Check trajectory indices
    traj_indices_batch = batch.get_trajectory_indices()
    assert torch.equal(traj_indices_batch, tlong(traj_indices, device=batch.device))
    # Check state indices
    state_indices_batch = batch.get_state_indices()
    assert torch.equal(state_indices_batch, tlong(state_indices, device=batch.device))
    # Check states
    states_batch = batch.get_states()
    states_policy_batch = batch.get_states(policy=True)
    if torch.is_tensor(states[0]):
        assert torch.equal(torch.stack(states_batch), torch.stack(states))
    else:
        assert states_batch == states
    assert torch.equal(states_policy_batch, env.states2policy(states))
    # Check actions
    actions_batch = batch.get_actions()
    assert torch.equal(
        actions_batch, tfloat(actions, float_type=batch.float, device=batch.device)
    )
    # Check done
    done_batch = batch.get_done()
    assert torch.equal(done_batch, tbool(dones, device=batch.device))
    # Check masks forward
    masks_forward_batch = batch.get_masks_forward()
    assert torch.equal(masks_forward_batch, tbool(masks_forward, device=batch.device))
    # Check masks parents forward
    masks_parents_forward_batch = batch.get_masks_forward(of_parents=True)
    assert torch.equal(
        masks_parents_forward_batch, tbool(masks_parents_forward, device=batch.device)
    )
    # Check masks backward
    masks_backward_batch = batch.get_masks_backward()
    assert torch.equal(masks_backward_batch, tbool(masks_backward, device=batch.device))
    # Check parents
    parents_batch = batch.get_parents()
    parents_policy_batch = batch.get_parents(policy=True)
    if torch.is_tensor(parents[0]):
        assert torch.equal(torch.stack(parents_batch), torch.stack(parents))
    else:
        assert parents_batch == parents
    assert torch.equal(parents_policy_batch, env.states2policy(parents))
    # Check parents_all
    if not env.continuous:
        parents_all_batch, parents_all_a_batch, _ = batch.get_parents_all()
        parents_all_policy_batch, _, _ = batch.get_parents_all(policy=True)
        if torch.is_tensor(parents_all[0]):
            assert torch.equal(torch.stack(parents_all_batch), torch.stack(parents_all))
        else:
            assert parents_all_batch == parents_all
        assert torch.equal(
            parents_all_a_batch,
            tfloat(
                parents_all_a,
                device=batch.device,
                float_type=batch.float,
            ),
        )
        assert torch.equal(parents_all_policy_batch, env.states2policy(parents_all))
    # Check rewards
    rewards_batch = batch.get_rewards()
    rewards = torch.stack(rewards)
    assert torch.all(
        torch.isclose(
            rewards_batch,
            tfloat(rewards, device=batch.device, float_type=batch.float),
        )
    ), (rewards, rewards_batch)
    # Check proxy values
    proxy_values_batch = batch.get_proxy_values()
    proxy_values = torch.stack(proxy_values)
    assert torch.all(
        torch.isclose(
            proxy_values_batch,
            tfloat(proxy_values, device=batch.device, float_type=batch.float),
        )
    ), (proxy_values, proxy_values_batch)
    # Check terminating states (sorted by trajectory)
    states_term_batch = batch.get_terminating_states(sort_by="traj")
    states_term_policy_batch = batch.get_terminating_states(sort_by="traj", policy=True)
    if torch.is_tensor(states_term_sorted[0]):
        assert torch.equal(
            torch.stack(states_term_batch), torch.stack(states_term_sorted)
        )
    else:
        assert states_term_batch == states_term_sorted
    assert torch.equal(states_term_policy_batch, env.states2policy(states_term_sorted))


@pytest.mark.repeat(N_REPETITIONS)
@pytest.mark.parametrize("env", ["grid2d", "tetris6x4", "ctorus2d5l"])
# @pytest.mark.skip(reason="skip while developping other tests")
def test__make_indices_consecutive__shuffled_indices_become_consecutive(
    env, batch, request
):
    batch_size = BATCH_SIZE
    env_ref = request.getfixturevalue(env)
    batch.set_env(env_ref)

    # Make list of envs
    envs = []
    shuffled2consecutive_dict = {}
    for consecutive_idx, shuffled_idx in enumerate(np.random.permutation(batch_size)):
        shuffled2consecutive_dict[shuffled_idx] = consecutive_idx
        env_aux = env_ref.copy().reset(shuffled_idx)
        envs.append(env_aux)

    # Initialize empty lists for checks
    traj_indices_shuffled = []
    traj_indices_consecutive = []

    # Iterate until envs is empty
    while envs:
        actions_iter = []
        valids_iter = []
        # Make step env by env (different to GFN Agent) to have full control
        for env in envs:
            parent = copy(env.state)
            # Sample random action
            state, action, valid = env.step_random()
            if valid:
                # Add to iter lists
                actions_iter.append(action)
                valids_iter.append(valid)
                # Add to checking lists
                traj_indices_shuffled.append(env.id)
                traj_indices_consecutive.append(shuffled2consecutive_dict[env.id])
        # Add all envs, actions and valids to batch
        batch.add_to_batch(envs, actions_iter, valids_iter)
        # Remove done envs
        envs = [env for env in envs if not env.done]

    # Check trajectory indices before making consecutive
    traj_indices_batch = batch.get_trajectory_indices()
    assert torch.equal(
        traj_indices_batch, tlong(traj_indices_shuffled, device=batch.device)
    )

    # Make consecutive
    batch.make_indices_consecutive()

    # Naively check that batch.trajectories and batch.envs keys are consecutive
    for idx, (traj_idx, env_idx) in enumerate(zip(batch.trajectories, batch.envs)):
        assert idx == traj_idx
        assert idx == env_idx
    # Check trajectory indices after making consecutive
    traj_indices_batch = batch.get_trajectory_indices()
    assert torch.equal(
        traj_indices_batch, tlong(traj_indices_consecutive, device=batch.device)
    )


@pytest.mark.repeat(N_REPETITIONS)
@pytest.mark.parametrize("env", ["grid2d", "tetris6x4", "ctorus2d5l"])
# @pytest.mark.skip(reason="skip while developping other tests")
def test__make_indices_consecutive__random_indices_become_consecutive(
    env, batch, request
):
    batch_size = BATCH_SIZE
    env_ref = request.getfixturevalue(env)
    batch.set_env(env_ref)

    # Make list of envs
    envs = []
    random2consecutive_dict = {}
    for consecutive_idx, random_idx in enumerate(
        np.random.permutation(batch_size * 10)[:batch_size]
    ):
        random2consecutive_dict[random_idx] = consecutive_idx
        env_aux = env_ref.copy().reset(random_idx)
        envs.append(env_aux)

    # Initialize empty lists for checks
    traj_indices_random = []
    traj_indices_consecutive = []

    # Iterate until envs is empty
    while envs:
        actions_iter = []
        valids_iter = []
        # Make step env by env (different to GFN Agent) to have full control
        for env in envs:
            parent = copy(env.state)
            # Sample random action
            state, action, valid = env.step_random()
            if valid:
                # Add to iter lists
                actions_iter.append(action)
                valids_iter.append(valid)
                # Add to checking lists
                traj_indices_random.append(env.id)
                traj_indices_consecutive.append(random2consecutive_dict[env.id])
        # Add all envs, actions and valids to batch
        batch.add_to_batch(envs, actions_iter, valids_iter)
        # Remove done envs
        envs = [env for env in envs if not env.done]

    # Check trajectory indices before making consecutive
    traj_indices_batch = batch.get_trajectory_indices()
    assert torch.equal(
        traj_indices_batch, tlong(traj_indices_random, device=batch.device)
    )

    # Make consecutive
    batch.make_indices_consecutive()

    # Naively check that batch.trajectories and batch.envs keys are consecutive
    for idx, (traj_idx, env_idx) in enumerate(zip(batch.trajectories, batch.envs)):
        assert idx == traj_idx
        assert idx == env_idx
    # Check trajectory indices after making consecutive
    traj_indices_batch = batch.get_trajectory_indices()
    assert torch.equal(
        traj_indices_batch, tlong(traj_indices_consecutive, device=batch.device)
    )


@pytest.mark.repeat(N_REPETITIONS)
@pytest.mark.parametrize("env", ["grid2d", "tetris6x4", "ctorus2d5l"])
# @pytest.mark.skip(reason="skip while developping other tests")
def test__make_indices_consecutive__multiplied_indices_become_consecutive(
    env, batch, request
):
    batch_size = BATCH_SIZE
    env_ref = request.getfixturevalue(env)
    batch.set_env(env_ref)

    # Make list of envs
    envs = []
    multiplied2consecutive_dict = {}
    for consecutive_idx in range(batch_size):
        multiplied_idx = consecutive_idx * 10
        multiplied2consecutive_dict[multiplied_idx] = consecutive_idx
        env_aux = env_ref.copy().reset(multiplied_idx)
        envs.append(env_aux)

    # Initialize empty lists for checks
    traj_indices_multiplied = []
    traj_indices_consecutive = []

    # Iterate until envs is empty
    while envs:
        actions_iter = []
        valids_iter = []
        # Make step env by env (different to GFN Agent) to have full control
        for env in envs:
            parent = copy(env.state)
            # Sample random action
            state, action, valid = env.step_random()
            if valid:
                # Add to iter lists
                actions_iter.append(action)
                valids_iter.append(valid)
                # Add to checking lists
                traj_indices_multiplied.append(env.id)
                traj_indices_consecutive.append(multiplied2consecutive_dict[env.id])
        # Add all envs, actions and valids to batch
        batch.add_to_batch(envs, actions_iter, valids_iter)
        # Remove done envs
        envs = [env for env in envs if not env.done]

    # Check trajectory indices before making consecutive
    traj_indices_batch = batch.get_trajectory_indices()
    assert torch.equal(
        traj_indices_batch, tlong(traj_indices_multiplied, device=batch.device)
    )

    # Make consecutive
    batch.make_indices_consecutive()

    # Naively check that batch.trajectories and batch.envs keys are consecutive
    for idx, (traj_idx, env_idx) in enumerate(zip(batch.trajectories, batch.envs)):
        assert idx == traj_idx
        assert idx == env_idx
    # Check trajectory indices after making consecutive
    traj_indices_batch = batch.get_trajectory_indices()
    assert torch.equal(
        traj_indices_batch, tlong(traj_indices_consecutive, device=batch.device)
    )


@pytest.mark.repeat(N_REPETITIONS)
@pytest.mark.parametrize(
    "env, proxy",
    [("grid2d", "corners"), ("tetris6x4", "tetris_score"), ("ctorus2d5l", "corners")],
)
def test__get_rewards__single_env_returns_expected_non_terminating(
    env, proxy, batch, request
):
    env = request.getfixturevalue(env)
    proxy = request.getfixturevalue(proxy)
    proxy.setup(env)
    env = env.reset()
    batch.set_env(env)
    batch.set_proxy(proxy)

    rewards = []
    while not env.done:
        parent = env.state
        # Sample random action
        _, action, valid = env.step_random()
        # Add to batch
        batch.add_to_batch([env], [action], [valid])
        if valid:
            rewards.append(proxy.rewards(env.state2proxy())[0])
    rewards_batch = batch.get_rewards(do_non_terminating=True)
    rewards = torch.stack(rewards)
    assert torch.all(
        torch.isclose(
            rewards_batch,
            tfloat(rewards, device=batch.device, float_type=batch.float),
        )
    ), (rewards, rewards_batch)


@pytest.mark.repeat(N_REPETITIONS)
@pytest.mark.parametrize(
    "env, proxy",
    [("grid2d", "corners"), ("tetris6x4", "tetris_score"), ("ctorus2d5l", "corners")],
)
def test__get_proxy_values__single_env_returns_expected_non_terminating(
    env, proxy, batch, request
):
    env = request.getfixturevalue(env)
    proxy = request.getfixturevalue(proxy)
    proxy.setup(env)
    env = env.reset()
    batch.set_env(env)
    batch.set_proxy(proxy)

    proxy_values = []
    while not env.done:
        parent = env.state
        # Sample random action
        _, action, valid = env.step_random()
        # Add to batch
        batch.add_to_batch([env], [action], [valid])
        if valid:
            proxy_values.append(proxy(env.state2proxy())[0])
    proxy_values_batch = batch.get_proxy_values(do_non_terminating=True)
    proxy_values = torch.stack(proxy_values)
    assert torch.all(
        torch.isclose(
            proxy_values_batch,
            tfloat(proxy_values, device=batch.device, float_type=batch.float),
        )
    ), (proxy_values, proxy_values_batch)


@pytest.mark.repeat(N_REPETITIONS)
@pytest.mark.parametrize(
    "env, proxy",
    [("grid2d", "corners"), ("tetris6x4", "tetris_score"), ("ctorus2d5l", "corners")],
)
def test__get_logrewards__single_env_returns_expected_non_terminating(
    env, proxy, batch, request
):
    env = request.getfixturevalue(env)
    proxy = request.getfixturevalue(proxy)
    proxy.setup(env)
    env = env.reset()
    batch.set_env(env)
    batch.set_proxy(proxy)

    logrewards = []
    while not env.done:
        parent = env.state
        # Sample random action
        _, action, valid = env.step_random()
        # Add to batch
        batch.add_to_batch([env], [action], [valid])
        if valid:
            logrewards.append(proxy.rewards(env.state2proxy(), log=True)[0])
    logrewards_batch = batch.get_rewards(log=True, do_non_terminating=True)
    logrewards = torch.stack(logrewards)
    assert torch.all(
        torch.isclose(
            logrewards_batch,
            tfloat(logrewards, device=batch.device, float_type=batch.float),
        )
    ), (logrewards, logrewards_batch)


@pytest.mark.repeat(N_REPETITIONS)
@pytest.mark.parametrize(
    "env, proxy",
    [("grid2d", "corners"), ("tetris6x4", "tetris_score_norm")],
)
def test__get_rewards_multiple_env_returns_expected_non_zero_non_terminating(
    env, proxy, batch, request
):
    batch_size = BATCH_SIZE
    env_ref = request.getfixturevalue(env)
    proxy = request.getfixturevalue(proxy)
    proxy.setup(env_ref)
    env_ref = env_ref.reset()

    batch.set_env(env_ref)
    batch.set_proxy(proxy)

    # Make list of envs
    envs = []
    for idx in range(batch_size):
        env_aux = env_ref.copy().reset(idx)
        envs.append(env_aux)

    rewards = []

    # Iterate until envs is empty
    while envs:
        actions_iter = []
        valids_iter = []
        # Make step env by env (different to GFN Agent) to have full control
        for env in envs:
            parent = copy(env.state)
            # Sample random action
            state, action, valid = env.step_random()
            if valid:
                # Add to iter lists
                actions_iter.append(action)
                valids_iter.append(valid)
                rewards.append(proxy.rewards(env.state2proxy())[0])
        # Add all envs, actions and valids to batch
        batch.add_to_batch(envs, actions_iter, valids_iter)
        # Remove done envs
        envs = [env for env in envs if not env.done]

    rewards_batch = batch.get_rewards(do_non_terminating=True)
    rewards = torch.stack(rewards)
    assert torch.all(
        torch.isclose(
            rewards_batch,
            tfloat(rewards, device=batch.device, float_type=batch.float),
        )
    ), (rewards, rewards_batch)
    assert ~torch.any(
        torch.isclose(rewards_batch, torch.zeros_like(rewards_batch))
    ), rewards_batch


@pytest.mark.repeat(N_REPETITIONS)
@pytest.mark.parametrize(
    "env, proxy",
    [("grid2d", "corners"), ("tetris6x4", "tetris_score_norm")],
)
def test__get_proxy_values_multiple_env_returns_expected_non_zero_non_terminating(
    env, proxy, batch, request
):
    batch_size = BATCH_SIZE
    env_ref = request.getfixturevalue(env)
    proxy = request.getfixturevalue(proxy)
    proxy.setup(env_ref)
    env_ref = env_ref.reset()

    batch.set_env(env_ref)
    batch.set_proxy(proxy)

    # Make list of envs
    envs = []
    for idx in range(batch_size):
        env_aux = env_ref.copy().reset(idx)
        envs.append(env_aux)

    proxy_values = []

    # Iterate until envs is empty
    while envs:
        actions_iter = []
        valids_iter = []
        # Make step env by env (different to GFN Agent) to have full control
        for env in envs:
            parent = copy(env.state)
            # Sample random action
            state, action, valid = env.step_random()
            if valid:
                # Add to iter lists
                actions_iter.append(action)
                valids_iter.append(valid)
                proxy_values.append(proxy(env.state2proxy())[0])
        # Add all envs, actions and valids to batch
        batch.add_to_batch(envs, actions_iter, valids_iter)
        # Remove done envs
        envs = [env for env in envs if not env.done]

    proxy_values_batch = batch.get_proxy_values(do_non_terminating=True)
    proxy_values = torch.stack(proxy_values)
    assert torch.all(
        torch.isclose(
            proxy_values_batch,
            tfloat(proxy_values, device=batch.device, float_type=batch.float),
        )
    ), (proxy_values, proxy_values_batch)
    assert ~torch.any(
        torch.isclose(proxy_values_batch, torch.zeros_like(proxy_values_batch))
    ), proxy_values_batch


@pytest.mark.repeat(N_REPETITIONS)
@pytest.mark.parametrize(
    "env, proxy",
    [("grid2d", "corners"), ("tetris6x4", "tetris_score_norm")],
)
def test__get_logrewards_multiple_env_returns_expected_non_zero_non_terminating(
    env, proxy, batch, request
):
    batch_size = BATCH_SIZE
    env_ref = request.getfixturevalue(env)
    proxy = request.getfixturevalue(proxy)
    proxy.setup(env_ref)
    env_ref = env_ref.reset()

    batch.set_env(env_ref)
    batch.set_proxy(proxy)

    # Make list of envs
    envs = []
    for idx in range(batch_size):
        env_aux = env_ref.copy().reset(idx)
        envs.append(env_aux)

    logrewards = []

    # Iterate until envs is empty
    while envs:
        actions_iter = []
        valids_iter = []
        # Make step env by env (different to GFN Agent) to have full control
        for env in envs:
            parent = copy(env.state)
            # Sample random action
            state, action, valid = env.step_random()
            if valid:
                # Add to iter lists
                actions_iter.append(action)
                valids_iter.append(valid)
                logrewards.append(proxy.rewards(env.state2proxy(), log=True)[0])
        # Add all envs, actions and valids to batch
        batch.add_to_batch(envs, actions_iter, valids_iter)
        # Remove done envs
        envs = [env for env in envs if not env.done]

    logrewards_batch = batch.get_rewards(log=True, do_non_terminating=True)
    logrewards = torch.stack(logrewards)
    assert torch.all(
        torch.isclose(
            logrewards_batch,
            tfloat(logrewards, device=batch.device, float_type=batch.float),
        )
    ), (logrewards, logrewards_batch)
    assert ~torch.any(
        torch.isclose(logrewards_batch, torch.zeros_like(logrewards_batch))
    ), logrewards_batch


@pytest.mark.repeat(N_REPETITIONS)
# @pytest.mark.skip(reason="skip while developping other tests")
@pytest.mark.parametrize(
    "env, proxy",
    [
        ("grid2d", "corners"),
        ("tetris6x4", "tetris_score_norm"),
        ("ctorus2d5l", "corners"),
    ],
)
def test__get_rewards_parents_multiple_env_returns_expected_non_terminating(
    env, proxy, batch, request
):
    batch_size = BATCH_SIZE
    env_ref = request.getfixturevalue(env)
    proxy = request.getfixturevalue(proxy)
    proxy.setup(env_ref)
    env_ref = env_ref.reset()

    batch.set_env(env_ref)
    batch.set_proxy(proxy)

    # Make list of envs
    envs = []
    for idx in range(batch_size):
        env_aux = env_ref.copy().reset(idx)
        envs.append(env_aux)

    rewards_parents = []
    rewards = []

    # Iterate until envs is empty
    while envs:
        actions_iter = []
        valids_iter = []
        # Make step env by env (different to GFN Agent) to have full control
        for env in envs:
            parent = copy(env.state)
            assert env.done is False

            # Sample random action
            state, action, valid = env.step_random()
            if valid:
                # Add to iter lists
                actions_iter.append(action)
                valids_iter.append(valid)
                rewards_parents.append(proxy.rewards(env.states2proxy([parent]))[0])
                rewards.append(proxy.rewards(env.state2proxy())[0])
        # Add all envs, actions and valids to batch
        batch.add_to_batch(envs, actions_iter, valids_iter)
        # Remove done envs
        envs = [env for env in envs if not env.done]

    rewards_parents_batch = batch.get_rewards_parents()
    rewards_parents = torch.stack(rewards_parents)

    rewards_batch = batch.get_rewards(do_non_terminating=True)
    rewards = torch.stack(rewards)

    assert torch.all(
        torch.isclose(
            rewards_parents_batch,
            tfloat(rewards_parents, device=batch.device, float_type=batch.float),
        )
    ), (rewards_parents, rewards_parents_batch)

    assert torch.all(
        torch.isclose(
            rewards_batch,
            tfloat(rewards, device=batch.device, float_type=batch.float),
        )
    ), (rewards, rewards_batch)


@pytest.mark.repeat(N_REPETITIONS)
# @pytest.mark.skip(reason="skip while developping other tests")
@pytest.mark.parametrize(
    "env, proxy",
    [
        ("grid2d", "corners"),
        ("tetris6x4", "tetris_score_norm"),
        ("ctorus2d5l", "corners"),
    ],
)
def test__get_logrewards_parents_multiple_env_returns_expected_non_terminating(
    env, proxy, batch, request
):
    batch_size = BATCH_SIZE
    env_ref = request.getfixturevalue(env)
    proxy = request.getfixturevalue(proxy)
    proxy.setup(env_ref)
    env_ref = env_ref.reset()

    batch.set_env(env_ref)
    batch.set_proxy(proxy)

    # Make list of envs
    envs = []
    for idx in range(batch_size):
        env_aux = env_ref.copy().reset(idx)
        envs.append(env_aux)

    logrewards_parents = []
    logrewards = []

    # Iterate until envs is empty
    while envs:
        actions_iter = []
        valids_iter = []
        # Make step env by env (different to GFN Agent) to have full control
        for env in envs:
            parent = copy(env.state)
            assert env.done is False

            # Sample random action
            state, action, valid = env.step_random()
            if valid:
                # Add to iter lists
                actions_iter.append(action)
                valids_iter.append(valid)
                logrewards_parents.append(
                    proxy.rewards(env.states2proxy([parent]), log=True)[0]
                )
                logrewards.append(proxy.rewards(env.state2proxy(), log=True)[0])
        # Add all envs, actions and valids to batch
        batch.add_to_batch(envs, actions_iter, valids_iter)
        # Remove done envs
        envs = [env for env in envs if not env.done]

    logrewards_parents_batch = batch.get_rewards_parents(log=True)
    logrewards_parents = torch.stack(logrewards_parents)

    logrewards_batch = batch.get_rewards(log=True, do_non_terminating=True)
    logrewards = torch.stack(logrewards)

    assert torch.all(
        torch.isclose(
            logrewards_parents_batch,
            tfloat(logrewards_parents, device=batch.device, float_type=batch.float),
        )
    ), (logrewards_parents, logrewards_parents_batch)

    assert torch.all(
        torch.isclose(
            logrewards_batch,
            tfloat(logrewards, device=batch.device, float_type=batch.float),
        )
    ), (logrewards, logrewards_batch)
