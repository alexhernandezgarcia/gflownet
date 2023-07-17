import pytest
import torch

from gflownet.envs.ctorus import ContinuousTorus
from gflownet.envs.grid import Grid
from gflownet.envs.tetris import Tetris
from gflownet.proxy.corners import Corners
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


@pytest.fixture
def batch():
    return Batch()


@pytest.fixture
def grid2d():
    return Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0)


#     return Grid(n_dim=5, length=10, cell_min=-1.0, cell_max=1.0)


@pytest.fixture
def tetris6x4():
    return Tetris(width=6, height=4, device="cpu")


#     return Tetris(width=10, height=20)


@pytest.fixture
def ctorus2d5l():
    return ContinuousTorus(n_dim=2, length_traj=10, n_comp=2)


#     return ContinuousTorus(n_dim=5, length_traj=10, n_comp=2)


@pytest.fixture()
def corners():
    return Corners(device="cpu", float_precision=32, mu=0.75, sigma=0.05)


@pytest.fixture()
def tetris_score():
    return TetrisScore(device="cpu", float_precision=32, normalize=False)


# @pytest.mark.skip(reason="skip while developping other tests")
def test__len__returnszero_at_init(batch):
    assert len(batch) == 0


@pytest.mark.repeat(10)
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


@pytest.mark.repeat(10)
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
    assert torch.equal(
        states_policy_batch,
        tfloat(
            env.statebatch2policy(states),
            device=batch.device,
            float_type=batch.float,
        ),
    )


@pytest.mark.repeat(10)
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
    assert torch.equal(
        parents_policy_batch,
        tfloat(
            env.statebatch2policy(parents),
            device=batch.device,
            float_type=batch.float,
        ),
    )


@pytest.mark.repeat(10)
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
    assert torch.equal(
        parents_all_policy_batch,
        tfloat(
            env.statebatch2policy(parents_all),
            device=batch.device,
            float_type=batch.float,
        ),
    )


@pytest.mark.repeat(10)
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


@pytest.mark.repeat(10)
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


@pytest.mark.repeat(10)
@pytest.mark.parametrize(
    "env, proxy",
    [("grid2d", "corners"), ("tetris6x4", "tetris_score"), ("ctorus2d5l", "corners")],
)
# @pytest.mark.skip(reason="skip while developping other tests")
def test__get_rewards__single_env_returns_expected(env, proxy, batch, request):
    env = request.getfixturevalue(env)
    proxy = request.getfixturevalue(proxy)
    env = env.reset()
    env.proxy = proxy
    env.setup_proxy()
    batch.set_env(env)

    rewards = []
    while not env.done:
        parent = env.state
        # Sample random action
        _, action, valid = env.step_random()
        # Add to batch
        batch.add_to_batch([env], [action], [valid])
        if valid:
            rewards.append(env.reward())
    rewards_batch = batch.get_rewards()
    rewards = torch.stack(rewards)
    assert torch.equal(
        rewards_batch,
        tfloat(rewards, device=batch.device, float_type=batch.float),
    ), (rewards, rewards_batch)


@pytest.mark.repeat(10)
@pytest.mark.parametrize(
    "env, proxy",
    [("grid2d", "corners"), ("tetris6x4", "tetris_score"), ("ctorus2d5l", "corners")],
)
# @pytest.mark.skip(reason="skip while developping other tests")
def test__forward_sampling_multiple_envs_all_as_expected(env, proxy, batch, request):
    batch_size = 10
    env_ref = request.getfixturevalue(env)
    proxy = request.getfixturevalue(proxy)
    env_ref.proxy = proxy
    env_ref.setup_proxy()
    batch.set_env(env_ref)

    # Make list of envs
    envs = []
    for idx in range(batch_size):
        env_aux = env_ref.copy().reset(idx)
        env_aux.proxy = proxy
        env_aux.setup_proxy()
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
                rewards.append(env.reward())
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
    assert torch.equal(
        states_policy_batch,
        tfloat(
            env.statebatch2policy(states),
            device=batch.device,
            float_type=batch.float,
        ),
    )
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
    assert torch.equal(
        parents_policy_batch,
        tfloat(
            env.statebatch2policy(parents),
            device=batch.device,
            float_type=batch.float,
        ),
    )
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
        assert torch.equal(
            parents_all_policy_batch,
            tfloat(
                env.statebatch2policy(parents_all),
                device=batch.device,
                float_type=batch.float,
            ),
        )
    # Check rewards
    rewards_batch = batch.get_rewards()
    rewards = torch.stack(rewards)
    assert torch.equal(
        rewards_batch,
        tfloat(rewards, device=batch.device, float_type=batch.float),
    ), (rewards, rewards_batch)
    # Check terminating states (sorted by trajectory)
    states_term_batch = batch.get_terminating_states(sort_by="traj")
    states_term_policy_batch = batch.get_terminating_states(sort_by="traj", policy=True)
    if torch.is_tensor(states_term_sorted[0]):
        assert torch.equal(
            torch.stack(states_term_batch), torch.stack(states_term_sorted)
        )
    else:
        assert states_term_batch == states_term_sorted
    assert torch.equal(
        states_term_policy_batch,
        tfloat(
            env.statebatch2policy(states_term_sorted),
            device=batch.device,
            float_type=batch.float,
        ),
    )


@pytest.mark.repeat(10)
@pytest.mark.parametrize(
    "env, proxy",
    [("grid2d", "corners"), ("tetris6x4", "tetris_score")],
)
# @pytest.mark.skip(reason="skip while developping other tests")
def test__backward_sampling_multiple_envs_all_as_expected(env, proxy, batch, request):
    batch_size = 10
    env_ref = request.getfixturevalue(env)
    proxy = request.getfixturevalue(proxy)
    env_ref.proxy = proxy
    env_ref.setup_proxy()
    batch.set_env(env_ref)

    # Sample terminating states and build list of envs
    x_batch = env_ref.get_uniform_terminating_states(n_states=batch_size)
    envs = []
    for idx, x in enumerate(x_batch):
        env_aux = env_ref.copy().reset(idx)
        env_aux = env_aux.set_state(state=x, done=True)
        env_aux.n_actions = env_aux.get_max_traj_length()
        env_aux.proxy = proxy
        env_aux.setup_proxy()
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
            reward = env.reward()
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
    assert torch.equal(
        states_policy_batch,
        tfloat(
            env.statebatch2policy(states),
            device=batch.device,
            float_type=batch.float,
        ),
    )
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
    assert torch.equal(
        parents_policy_batch,
        tfloat(
            env.statebatch2policy(parents),
            device=batch.device,
            float_type=batch.float,
        ),
    )
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
        assert torch.equal(
            parents_all_policy_batch,
            tfloat(
                env.statebatch2policy(parents_all),
                device=batch.device,
                float_type=batch.float,
            ),
        )
    # Check rewards
    rewards_batch = batch.get_rewards()
    rewards = torch.stack(rewards)
    assert torch.equal(
        rewards_batch,
        tfloat(rewards, device=batch.device, float_type=batch.float),
    ), (rewards, rewards_batch)
    # Check terminating states (sorted by trajectory)
    states_term_batch = batch.get_terminating_states(sort_by="traj")
    states_term_policy_batch = batch.get_terminating_states(sort_by="traj", policy=True)
    if torch.is_tensor(states_term_sorted[0]):
        assert torch.equal(
            torch.stack(states_term_batch), torch.stack(states_term_sorted)
        )
    else:
        assert states_term_batch == states_term_sorted
    assert torch.equal(
        states_term_policy_batch,
        tfloat(
            env.statebatch2policy(states_term_sorted),
            device=batch.device,
            float_type=batch.float,
        ),
    )


@pytest.mark.repeat(10)
@pytest.mark.parametrize(
    "env, proxy",
    [("grid2d", "corners"), ("tetris6x4", "tetris_score")],
)
# @pytest.mark.skip(reason="skip while developping other tests")
def test__mixed_sampling_multiple_envs_all_as_expected(env, proxy, batch, request):
    # Initialize fixtures and batch
    env_ref = request.getfixturevalue(env)
    proxy = request.getfixturevalue(proxy)
    env_ref.proxy = proxy
    env_ref.setup_proxy()
    batch.set_env(env_ref)

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
    traj_indices = []
    state_indices = []
    states_term_sorted = []

    ### FORWARD ###

    # Make list of envs
    batch_size_forward = 10
    envs = []
    for idx in range(batch_size_forward):
        env_aux = env_ref.copy().reset(idx)
        env_aux.proxy = proxy
        env_aux.setup_proxy()
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
                rewards.append(env.reward())
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
    batch_size_backward = 10
    x_batch = env_ref.get_uniform_terminating_states(n_states=batch_size_backward)
    envs = []
    for idx, x in enumerate(x_batch):
        env_aux = env_ref.copy().reset(idx + batch_size_forward)
        env_aux = env_aux.set_state(state=x, done=True)
        env_aux.n_actions = env_aux.get_max_traj_length()
        env_aux.proxy = proxy
        env_aux.setup_proxy()
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
            reward = env.reward()
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
    assert torch.equal(
        states_policy_batch,
        tfloat(
            env.statebatch2policy(states),
            device=batch.device,
            float_type=batch.float,
        ),
    )
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
    assert torch.equal(
        parents_policy_batch,
        tfloat(
            env.statebatch2policy(parents),
            device=batch.device,
            float_type=batch.float,
        ),
    )
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
        assert torch.equal(
            parents_all_policy_batch,
            tfloat(
                env.statebatch2policy(parents_all),
                device=batch.device,
                float_type=batch.float,
            ),
        )
    # Check rewards
    rewards_batch = batch.get_rewards()
    rewards = torch.stack(rewards)
    assert torch.equal(
        rewards_batch,
        tfloat(rewards, device=batch.device, float_type=batch.float),
    ), (rewards, rewards_batch)
    # Check terminating states (sorted by trajectory)
    states_term_batch = batch.get_terminating_states(sort_by="traj")
    states_term_policy_batch = batch.get_terminating_states(sort_by="traj", policy=True)
    if torch.is_tensor(states_term_sorted[0]):
        assert torch.equal(
            torch.stack(states_term_batch), torch.stack(states_term_sorted)
        )
    else:
        assert states_term_batch == states_term_sorted
    assert torch.equal(
        states_term_policy_batch,
        tfloat(
            env.statebatch2policy(states_term_sorted),
            device=batch.device,
            float_type=batch.float,
        ),
    )


@pytest.mark.repeat(10)
@pytest.mark.parametrize(
    "env, proxy",
    [("grid2d", "corners"), ("tetris6x4", "tetris_score")],
)
# @pytest.mark.skip(reason="skip while developping other tests")
def test__mixed_sampling_merged_all_as_expected(env, proxy, request):
    # Initialize fixtures and batch
    env_ref = request.getfixturevalue(env)
    proxy = request.getfixturevalue(proxy)
    env_ref.proxy = proxy
    env_ref.setup_proxy()
    batch_fw = Batch(env=env_ref)
    batch_bw = Batch(env=env_ref)

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
    traj_indices = []
    state_indices = []
    states_term_sorted = []

    ### FORWARD ###

    # Make list of envs
    batch_size_forward = 10
    envs = []
    for idx in range(batch_size_forward):
        env_aux = env_ref.copy().reset(idx)
        env_aux.proxy = proxy
        env_aux.setup_proxy()
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
                rewards.append(env.reward())
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
    batch_size_backward = 10
    x_batch = env_ref.get_uniform_terminating_states(n_states=batch_size_backward)
    envs = []
    for idx, x in enumerate(x_batch):
        env_aux = env_ref.copy().reset(idx)
        env_aux = env_aux.set_state(state=x, done=True)
        env_aux.n_actions = env_aux.get_max_traj_length()
        env_aux.proxy = proxy
        env_aux.setup_proxy()
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
            reward = env.reward()
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
                traj_indices.append(env.id + batch_size_forward)
                state_indices.append(env.n_actions)
        # Add all envs, actions and valids to batch
        batch_bw.add_to_batch(envs, actions_iter, valids_iter, backward=True)
        # Remove finished trajectories (state reached source)
        envs = [env for env in envs if not env.equal(env.state, env.source)]

    ### MERGE BATCHES ###

    batch = Batch(env=env_ref)
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
    assert torch.equal(
        states_policy_batch,
        tfloat(
            env.statebatch2policy(states),
            device=batch.device,
            float_type=batch.float,
        ),
    )
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
    assert torch.equal(
        parents_policy_batch,
        tfloat(
            env.statebatch2policy(parents),
            device=batch.device,
            float_type=batch.float,
        ),
    )
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
        assert torch.equal(
            parents_all_policy_batch,
            tfloat(
                env.statebatch2policy(parents_all),
                device=batch.device,
                float_type=batch.float,
            ),
        )
    # Check rewards
    rewards_batch = batch.get_rewards()
    rewards = torch.stack(rewards)
    assert torch.equal(
        rewards_batch,
        tfloat(rewards, device=batch.device, float_type=batch.float),
    ), (rewards, rewards_batch)
    # Check terminating states (sorted by trajectory)
    states_term_batch = batch.get_terminating_states(sort_by="traj")
    states_term_policy_batch = batch.get_terminating_states(sort_by="traj", policy=True)
    if torch.is_tensor(states_term_sorted[0]):
        assert torch.equal(
            torch.stack(states_term_batch), torch.stack(states_term_sorted)
        )
    else:
        assert states_term_batch == states_term_sorted
    assert torch.equal(
        states_term_policy_batch,
        tfloat(
            env.statebatch2policy(states_term_sorted),
            device=batch.device,
            float_type=batch.float,
        ),
    )
