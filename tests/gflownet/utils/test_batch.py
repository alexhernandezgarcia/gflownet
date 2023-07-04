import pytest
import torch

from gflownet.envs.grid import Grid
from gflownet.proxy.uniform import Uniform
from gflownet.proxy.corners import Corners
from gflownet.utils.batch import Batch
from gflownet.utils.common import (
    concat_items,
    set_device,
    set_float_precision,
    tbool,
    tfloat,
    tint,
    tlong,
)


@pytest.fixture
def batch_tb():
    return Batch(loss="trajectorybalance")


@pytest.fixture
def batch_fm():
    return Batch(loss="flowmatch")


@pytest.fixture
def grid2d():
    return Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0)


@pytest.fixture()
def uniform_proxy():
    return Uniform()


def test__len__returnszero_at_init(batch_tb, batch_fm):
    assert len(batch_tb) == 0
    assert len(batch_fm) == 0


@pytest.mark.repeat(10)
def test__add_to_batch__single_env_adds_expected(grid2d, batch_tb):
    grid2d = grid2d.reset()
    while not grid2d.done:
        # Sample random action
        state, action, valid = grid2d.step_random()
        # Add to batch
        batch_tb.add_to_batch([grid2d], [action], [valid])
        if valid is False:
            continue
        # Checks
        assert batch_tb.env_ids[-1] == grid2d.id
        assert batch_tb.states[-1] == state
        assert batch_tb.actions[-1] == action
        assert batch_tb.done[-1] == grid2d.done
        assert batch_tb.n_actions[-1] == grid2d.n_actions


@pytest.mark.repeat(10)
def test__get_states__single_env_returns_expected(grid2d, batch_tb):
    grid2d = grid2d.reset()
    states = []
    while not grid2d.done:
        # Sample random action
        state, action, valid = grid2d.step_random()
        # Add to batch
        batch_tb.add_to_batch([grid2d], [action], [valid])
        if valid:
            states.append(state)
    states_batch = batch_tb.get_states()
    states_policy_batch = batch_tb.get_states(policy=True)
    assert states_batch == states
    assert torch.equal(
        states_policy_batch,
        tfloat(
            grid2d.statebatch2policy(states),
            device=batch_tb.device,
            float_type=batch_tb.float,
        ),
    )


@pytest.mark.repeat(10)
def test__get_parents__single_env_returns_expected(grid2d, batch_tb):
    grid2d = grid2d.reset()
    parents = []
    while not grid2d.done:
        parent = grid2d.state
        # Sample random action
        _, action, valid = grid2d.step_random()
        # Add to batch
        batch_tb.add_to_batch([grid2d], [action], [valid])
        if valid:
            parents.append(parent)
    parents_batch = batch_tb.get_parents()
    parents_policy_batch = batch_tb.get_parents(policy=True)
    assert parents_batch == parents
    assert torch.equal(
        parents_policy_batch,
        tfloat(
            grid2d.statebatch2policy(parents),
            device=batch_tb.device,
            float_type=batch_tb.float,
        ),
    )


@pytest.mark.repeat(10)
def test__get_parents_all__single_env_returns_expected(grid2d, batch_tb):
    grid2d = grid2d.reset()
    parents_all = []
    parents_all_a = []
    while not grid2d.done:
        # Sample random action
        _, action, valid = grid2d.step_random()
        # Add to batch
        batch_tb.add_to_batch([grid2d], [action], [valid])
        if valid:
            parents, parents_a = grid2d.get_parents()
            parents_all.extend(parents)
            parents_all_a.extend(parents_a)
    parents_all_batch, parents_all_a_batch, _ = batch_tb.get_parents_all()
    parents_all_policy_batch, _, _ = batch_tb.get_parents_all(policy=True)
    assert parents_all_batch == parents_all
    assert torch.equal(
        parents_all_a_batch,
        tfloat(
            parents_all_a,
            device=batch_tb.device,
            float_type=batch_tb.float,
        ),
    )
    assert torch.equal(
        parents_all_policy_batch,
        tfloat(
            grid2d.statebatch2policy(parents_all),
            device=batch_tb.device,
            float_type=batch_tb.float,
        ),
    )


@pytest.mark.repeat(10)
def test__get_masks_forward__single_env_returns_expected(grid2d, batch_tb):
    grid2d = grid2d.reset()
    masks_forward = []
    while not grid2d.done:
        parent = grid2d.state
        # Sample random action
        _, action, valid = grid2d.step_random()
        # Add to batch
        batch_tb.add_to_batch([grid2d], [action], [valid])
        if valid:
            masks_forward.append(grid2d.get_mask_invalid_actions_forward())
    masks_forward_batch = batch_tb.get_masks_forward()
    assert torch.equal(
        masks_forward_batch, tbool(masks_forward, device=batch_tb.device)
    )


@pytest.mark.repeat(10)
def test__get_masks_backward__single_env_returns_expected(grid2d, batch_tb):
    grid2d = grid2d.reset()
    masks_backward = []
    while not grid2d.done:
        parent = grid2d.state
        # Sample random action
        _, action, valid = grid2d.step_random()
        # Add to batch
        batch_tb.add_to_batch([grid2d], [action], [valid])
        if valid:
            masks_backward.append(grid2d.get_mask_invalid_actions_backward())
    masks_backward_batch = batch_tb.get_masks_backward()
    assert torch.equal(
        masks_backward_batch, tbool(masks_backward, device=batch_tb.device)
    )


@pytest.mark.repeat(10)
def test__get_rewards__single_env_returns_expected(grid2d, batch_tb):
    grid2d = grid2d.reset()
    proxy = Corners(device=grid2d.device, float_precision=grid2d.float, mu=0.75, sigma=0.05)
    grid2d.proxy = proxy
    grid2d.setup_proxy()

    rewards = []
    while not grid2d.done:
        parent = grid2d.state
        # Sample random action
        _, action, valid = grid2d.step_random()
        # Add to batch
        batch_tb.add_to_batch([grid2d], [action], [valid])
        if valid:
            rewards.append(grid2d.reward())
    rewards_batch = batch_tb.get_rewards()
    assert torch.equal(
        rewards_batch, tfloat(rewards, device=batch_tb.device)
    )


@pytest.mark.parametrize(
    "action, state_expected",
    [
        (
            (1, 0),
            [1, 0],
        ),
        (
            (0, 1),
            [0, 1],
        ),
    ],
)
def test__add_to_batch__minimal_grid2d_returns_expected(
    batch_tb, batch_fm, grid2d, action, state_expected
):
    state, action_step, valid = grid2d.step(action)
    assert state == state_expected
    assert action_step == action
    assert valid
    # TB
    batch_tb.add_to_batch([grid2d], [action], [valid])
    assert batch_tb.states == [state_expected]
    assert batch_tb.actions == [action]
    # FM
    batch_fm.add_to_batch([grid2d], [action], [valid])
    assert batch_fm.states == [state_expected]
    assert batch_fm.actions == [action]
