import common
import pytest
import torch

from gflownet.envs.tetris import Tetris


@pytest.fixture
def env():
    return Tetris(width=4, height=5)


@pytest.mark.parametrize(
    "action_space",
    [
        [
            (1, 0, 0),
            (1, 0, 1),
            (1, 0, 2),
            (1, 0, 3),
            (1, 90, 0),
            (2, 0, 0),
            (2, 0, 1),
            (2, 0, 2),
            (2, 90, 0),
            (2, 90, 1),
            (2, 180, 0),
            (2, 180, 1),
            (2, 180, 2),
            (2, 270, 0),
            (2, 270, 1),
            (3, 0, 0),
            (3, 0, 1),
            (3, 0, 2),
            (3, 90, 0),
            (3, 90, 1),
            (3, 180, 0),
            (3, 180, 1),
            (3, 180, 2),
            (3, 270, 0),
            (3, 270, 1),
            (4, 0, 0),
            (4, 0, 1),
            (4, 0, 2),
            (5, 0, 0),
            (5, 0, 1),
            (5, 90, 0),
            (5, 90, 1),
            (5, 90, 2),
            (6, 0, 0),
            (6, 0, 1),
            (6, 90, 0),
            (6, 90, 1),
            (6, 90, 2),
            (6, 180, 0),
            (6, 180, 1),
            (6, 270, 0),
            (6, 270, 1),
            (6, 270, 2),
            (7, 0, 0),
            (7, 0, 1),
            (7, 90, 0),
            (7, 90, 1),
            (7, 90, 2),
            (-1, -1, -1),
        ],
    ],
)
def test__get_action_space__returns_expected(env, action_space):
    assert set(action_space) == set(env.action_space)


@pytest.mark.parametrize(
    "state, action, next_state",
    [
        (
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            (4, 0, 0),
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [4, 4, 0, 0], [4, 4, 0, 0]],
        ),
        (
            [[0, 0, 0, 0], [0, 0, 0, 0], [5, 0, 0, 0], [5, 5, 0, 0], [0, 5, 0, 0]],
            (5, 90, 1),
            [[0, 0, 0, 0], [0, 5, 0, 0], [5, 5, 5, 0], [5, 5, 5, 0], [0, 5, 0, 0]],
        ),
    ],
)
def test__step__returns_expected(env, state, action, next_state):
    env.set_state(torch.tensor(state, dtype=torch.uint8))
    env.step(action)
    assert torch.equal(env.state, torch.tensor(next_state, dtype=torch.uint8))
