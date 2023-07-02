import pytest
import torch

from gflownet.envs.grid import Grid
from gflownet.utils.batch import Batch


@pytest.fixture
def batch_tb():
    return Batch(loss="trajectorybalance")


@pytest.fixture
def batch_fm():
    return Batch(loss="flowmatch")


@pytest.fixture
def grid2d():
    return Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0)


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

# @pytest.mark.repeat(10)
# def test__get_parents__single_env_returns_expected(grid2d, batch_tb):
#     grid2d = grid2d.reset()
#     parents = []
#     while not grid2d.done:
#         parent = grid2d.state
#         # Sample random action
#         _, action, valid = grid2d.step_random()
#         # Add to batch
#         batch_tb.add_to_batch([grid2d], [action], [valid])
#         if valid:
#             parents.append(parent)
#     parents_batch = batch_tb.get_parents()
#     assert parents_batch == parents

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
    assert states_batch == states

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
