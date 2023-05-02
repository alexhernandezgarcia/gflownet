import common
import pytest
import torch

from gflownet.envs.tetris import Tetris


@pytest.fixture
def env():
    return Tetris(width=4, height=5)


@pytest.fixture
def env6x4():
    return Tetris(width=4, height=6)


@pytest.fixture
def env_mini():
    return Tetris(width=4, height=5, pieces=["I", "O"], rotations=[0])


@pytest.fixture
def env_1piece():
    return Tetris(width=4, height=5, pieces=["O"], rotations=[0])


@pytest.fixture
def env_full():
    return Tetris(width=10, height=20)


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
    "state, action, state_next_expected, valid_expected",
    [
        (
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            (4, 0, 0),
            [
                [000, 000, 000, 000],
                [000, 000, 000, 000],
                [000, 000, 000, 000],
                [400, 400, 000, 000],
                [400, 400, 000, 000],
            ],
            True,
        ),
        (
            [
                [000, 000, 000, 000],
                [000, 000, 000, 000],
                [000, 000, 000, 000],
                [400, 400, 000, 000],
                [400, 400, 000, 000],
            ],
            (4, 0, 0),
            [
                [000, 000, 000, 000],
                [401, 401, 000, 000],
                [401, 401, 000, 000],
                [400, 400, 000, 000],
                [400, 400, 000, 000],
            ],
            True,
        ),
        (
            [
                [000, 000, 000, 000],
                [000, 000, 000, 000],
                [000, 000, 600, 000],
                [400, 400, 600, 600],
                [400, 400, 600, 000],
            ],
            (4, 0, 0),
            [
                [000, 000, 000, 000],
                [401, 401, 000, 000],
                [401, 401, 600, 000],
                [400, 400, 600, 600],
                [400, 400, 600, 000],
            ],
            True,
        ),
        (
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            (1, 0, 3),
            [
                [000, 000, 000, 000],
                [000, 000, 000, 100],
                [000, 000, 000, 100],
                [000, 000, 000, 100],
                [000, 000, 000, 100],
            ],
            True,
        ),
        (
            [
                [000, 000, 000, 000],
                [402, 402, 403, 403],
                [402, 402, 403, 403],
                [400, 400, 401, 401],
                [400, 400, 401, 401],
            ],
            (1, 90, 0),
            [
                [100, 100, 100, 100],
                [402, 402, 403, 403],
                [402, 402, 403, 403],
                [400, 400, 401, 401],
                [400, 400, 401, 401],
            ],
            True,
        ),
        (
            [
                [100, 100, 100, 100],
                [402, 402, 403, 403],
                [402, 402, 403, 403],
                [400, 400, 401, 401],
                [400, 400, 401, 401],
            ],
            (1, 90, 0),
            [
                [100, 100, 100, 100],
                [402, 402, 403, 403],
                [402, 402, 403, 403],
                [400, 400, 401, 401],
                [400, 400, 401, 401],
            ],
            False,
        ),
    ],
)
def test__drop_piece_on_board__returns_expected(
    env, state, action, state_next_expected, valid_expected
):
    state = torch.tensor(state, dtype=torch.int16)
    state_next_expected = torch.tensor(state_next_expected, dtype=torch.int16)
    env.set_state(state)
    state_next, valid = env._drop_piece_on_board(action)
    assert torch.equal(state_next, state_next_expected)


@pytest.mark.parametrize(
    "state, mask_expected",
    [
        (
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            [False, False, False, True],
        ),
        (
            [
                [000, 000, 000, 000],
                [000, 000, 000, 000],
                [000, 000, 000, 000],
                [000, 400, 400, 000],
                [000, 400, 400, 000],
            ],
            [False, False, False, True],
        ),
        (
            [
                [000, 000, 000, 000],
                [000, 000, 000, 000],
                [000, 000, 000, 000],
                [400, 400, 400, 400],
                [400, 400, 400, 400],
            ],
            [False, False, False, True],
        ),
    ],
)
def test__mask_invalid_actions_forward__returns_expected(
    env_1piece, state, mask_expected
):
    state = torch.tensor(state, dtype=torch.int16)
    mask = env_1piece.get_mask_invalid_actions_forward(state, False)
    assert mask == mask_expected


@pytest.mark.parametrize(
    "state, action, next_state",
    [
        (
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            (4, 0, 0),
            [
                [000, 000, 000, 000],
                [000, 000, 000, 000],
                [000, 000, 000, 000],
                [400, 400, 000, 000],
                [400, 400, 000, 000],
            ],
        ),
        (
            [
                [000, 000, 000, 000],
                [000, 000, 000, 000],
                [500, 000, 000, 000],
                [500, 500, 000, 000],
                [000, 500, 000, 000],
            ],
            (5, 90, 1),
            [
                [000, 000, 000, 000],
                [000, 501, 000, 000],
                [500, 501, 501, 000],
                [500, 500, 501, 0],
                [000, 500, 000, 0],
            ],
        ),
    ],
)
def test__step__returns_expected(env, state, action, next_state):
    env.set_state(torch.tensor(state, dtype=torch.int16))
    env.step(action)
    assert torch.equal(env.state, torch.tensor(next_state, dtype=torch.int16))


@pytest.mark.parametrize(
    "board, piece_idx, expected",
    [
        (
            [
                [000, 000, 000, 000],
                [000, 000, 000, 000],
                [000, 000, 000, 000],
                [400, 400, 000, 000],
                [400, 400, 000, 000],
            ],
            4,
            True,
        ),
        (
            [
                [000, 000, 000, 000],
                [000, 501, 000, 000],
                [500, 501, 501, 000],
                [500, 500, 501, 000],
                [000, 500, 000, 000],
            ],
            500,
            False,
        ),
        (
            [
                [000, 000, 000, 000],
                [000, 000, 000, 000],
                [000, 600, 600, 600],
                [400, 400, 600, 000],
                [400, 400, 000, 000],
            ],
            400,
            False,
        ),
        (
            [
                [000, 000, 000, 000],
                [000, 000, 000, 000],
                [000, 600, 600, 600],
                [400, 400, 600, 000],
                [400, 400, 000, 000],
            ],
            4,
            False,
        ),
        (
            [
                [000, 000, 000, 000],
                [000, 000, 000, 000],
                [000, 600, 600, 600],
                [400, 400, 600, 000],
                [400, 400, 000, 000],
            ],
            6,
            True,
        ),
        (
            [
                [000, 000, 000, 000],
                [000, 000, 000, 000],
                [000, 600, 600, 600],
                [400, 400, 600, 000],
                [400, 400, 000, 000],
            ],
            600,
            True,
        ),
    ],
)
def test__piece_can_be_lifted__returns_expected(env, board, piece_idx, expected):
    board = torch.tensor(board, dtype=torch.int16)
    assert env._piece_can_be_lifted(board, piece_idx) == expected


@pytest.mark.parametrize(
    "state, parents_expected, parents_a_expected",
    [
        (
            [
                [100, 000, 000, 000],
                [100, 000, 000, 000],
                [100, 000, 000, 000],
                [100, 000, 000, 000],
                [400, 400, 401, 401],
                [400, 400, 401, 401],
            ],
            [
                [
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [400, 400, 401, 401],
                    [400, 400, 401, 401],
                ],
                [
                    [100, 000, 000, 000],
                    [100, 000, 000, 000],
                    [100, 000, 000, 000],
                    [100, 000, 000, 000],
                    [400, 400, 000, 000],
                    [400, 400, 000, 000],
                ],
            ],
            [(1, 0, 0), (4, 0, 2)],
        ),
        (
            [
                [100, 000, 000, 101],
                [100, 000, 000, 101],
                [100, 000, 000, 101],
                [100, 000, 000, 101],
                [400, 400, 401, 401],
                [400, 400, 401, 401],
            ],
            [
                [
                    [000, 000, 000, 101],
                    [000, 000, 000, 101],
                    [000, 000, 000, 101],
                    [000, 000, 000, 101],
                    [400, 400, 401, 401],
                    [400, 400, 401, 401],
                ],
                [
                    [100, 000, 000, 000],
                    [100, 000, 000, 000],
                    [100, 000, 000, 000],
                    [100, 000, 000, 000],
                    [400, 400, 401, 401],
                    [400, 400, 401, 401],
                ],
            ],
            [(1, 0, 0), (1, 0, 3)],
        ),
    ],
)
def test__get_parents__returns_expected(
    env6x4, state, parents_expected, parents_a_expected
):
    state = torch.tensor(state, dtype=torch.int16)
    parents_expected = [
        torch.tensor(parent, dtype=torch.int16) for parent in parents_expected
    ]
    parents, parents_a = env6x4.get_parents(state)
    for p, p_e in zip(parents, parents_expected):
        assert torch.equal(p, p_e)
    for p_a, p_a_e in zip(parents_a, parents_a_expected):
        assert p_a == p_a_e


#
#
@pytest.mark.parametrize(
    "state, parent_expected, parent_a_expected",
    [
        (
            [
                [000, 000, 000, 602],
                [000, 601, 602, 602],
                [601, 601, 601, 602],
                [000, 600, 600, 600],
                [000, 000, 600, 000],
            ],
            [
                [000, 000, 000, 000],
                [000, 601, 000, 000],
                [601, 601, 601, 000],
                [000, 600, 600, 600],
                [000, 000, 600, 000],
            ],
            (6, 270, 2),
        ),
        (
            [
                [101, 101, 101, 101],
                [100, 100, 100, 100],
                [000, 000, 200, 000],
                [000, 000, 200, 000],
                [000, 200, 200, 000],
            ],
            [
                [000, 000, 000, 000],
                [100, 100, 100, 100],
                [000, 000, 200, 000],
                [000, 000, 200, 000],
                [000, 200, 200, 000],
            ],
            (1, 90, 0),
        ),
    ],
)
def test__get_parents__contains_expected(
    env, state, parent_expected, parent_a_expected
):
    state = torch.tensor(state, dtype=torch.int16)
    parent_expected = torch.tensor(parent_expected, dtype=torch.int16)
    parents, parents_a = env.get_parents(state)
    assert any([torch.equal(p, parent_expected) for p in parents])
    assert any([a == parent_a_expected for a in parents_a])


def test__all_env_common(env_1piece):
    return common.test__all_env_common(env_1piece)


def test__all_env_common(env_mini):
    return common.test__all_env_common(env_mini)


def test__all_env_common(env):
    return common.test__all_env_common(env)


@pytest.mark.skip(reason="Takes too long")
def test__all_env_common(env_full):
    return common.test__all_env_common(env_full)
