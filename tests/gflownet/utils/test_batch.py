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
    return Tetris(width=6, height=4)


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
        assert batch.n_actions[-1] == env.n_actions


@pytest.mark.repeat(10)
@pytest.mark.parametrize("env", ["grid2d", "tetris6x4", "ctorus2d5l"])
# @pytest.mark.skip(reason="skip while developping other tests")
def test__get_states__single_env_returns_expected(env, batch, request):
    env = request.getfixturevalue(env)
    env = env.reset()
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
def test__multiple_envs_all_as_expected(env, proxy, batch, request):
    batch_size = 10
    env_ref = request.getfixturevalue(env)
    proxy = request.getfixturevalue(proxy)

    # Make list of envs
    envs = []
    for idx in range(batch_size):
        env_aux = env_ref.copy().reset()
        env_aux.proxy = proxy
        env_aux.setup_proxy()
        envs.append(env_aux)

    # Initialize empty lists for checks
    states = []
    actions = []
    done = []
    masks_forward = []
    masks_backward = []
    parents = []
    parents_all = []
    parents_all_a = []
    rewards = []
    traj_indices = []
    state_indices = []

    # Iterate until envs is empty
    while envs:
        # Make step env by env (different to GFN Agent) to have full control
        actions_iter = []
        valids_iter = []
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
                masks_backward.append(env.get_mask_invalid_actions_backward())
                parents.append(parent)
                if not env.continuous:
                    env_parents, env_parents_a = env.get_parents()
                    parents_all.extend(env_parents)
                    parents_all_a.extend(env_parents_a)
                rewards.append(env.reward())
                traj_indices.append(env.id)
                state_indices.append(env.n_actions)
        # Add all envs, actions and valids to batch
        batch.add_to_batch(envs, actions_iter, valids_iter)
        # Remove done envs
        envs = [env for env in envs if not env.done]

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
    # Check masks forward
    masks_forward_batch = batch.get_masks_forward()
    assert torch.equal(masks_forward_batch, tbool(masks_forward, device=batch.device))
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
