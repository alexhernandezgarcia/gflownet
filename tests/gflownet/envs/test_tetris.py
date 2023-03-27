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


@pytest.mark.parametrize(
    "board, piece_mat, row, col, expected",
    [
        (
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [4, 4, 0, 0], [4, 4, 0, 0]],
            [[4, 4], [4, 4]],
            3,
            0,
            True,
        ),
        (
            [[0, 0, 0, 0], [0, 5, 0, 0], [5, 5, 5, 0], [5, 5, 5, 0], [0, 5, 0, 0]],
            [[0, 5, 5], [5, 5, 0]],
            2,
            0,
            False,
        ),
        (
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 6, 6, 6], [4, 4, 6, 0], [4, 4, 0, 0]],
            [[4, 4], [4, 4]],
            3,
            0,
            False,
        ),
        (
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 6, 6, 6], [4, 4, 6, 0], [4, 4, 0, 0]],
            [[6, 6, 6], [0, 6, 0]],
            2,
            1,
            True,
        ),
        (
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 6, 6, 6], [4, 4, 6, 0], [4, 4, 0, 0]],
            [[6, 6, 6], [0, 6, 0]],
            3,
            1,
            False,
        ),
    ],
)
def test__piece_can_be_lifted__returns_expected(
    env, board, piece_mat, row, col, expected
):
    board = torch.tensor(board, dtype=torch.uint8)
    piece_mat = torch.tensor(piece_mat, dtype=torch.uint8)
    assert Tetris._piece_can_be_lifted(board, piece_mat, row, col) == expected


@pytest.mark.parametrize(
    "input, piece_idx, expected",
    [
        (
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [4, 4, 0, 0], [4, 4, 0, 0]],
            4,
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        ),
        (
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [4, 4, 4, 4], [4, 4, 4, 4]],
            4,
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        ),
        (
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [4, 4, 4, 0], [4, 4, 4, 0]],
            4,
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 4, 0]],
        ),
        (
            [[0, 0, 0, 0], [0, 5, 0, 0], [5, 5, 5, 0], [5, 5, 5, 0], [0, 5, 0, 0]],
            5,
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        ),
        (
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [4, 0, 0, 4], [4, 0, 0, 4]],
            4,
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [4, 0, 0, 4], [4, 0, 0, 4]],
        ),
    ],
)
def test__remove_all_pieces__returns_expected(env, input, piece_idx, expected):
    input = torch.tensor(input, dtype=torch.uint8)
    expected = torch.tensor(expected, dtype=torch.uint8)
    assert torch.equal(env._remove_all_pieces(input, piece_idx), expected)


@pytest.mark.parametrize(
    "board, action, expected",
    [
        (
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [4, 4, 0, 0], [4, 4, 0, 0]],
            (4, 0, 0),
            True,
        ),
        (
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [4, 4, 0, 0], [4, 4, 0, 0]],
            (1, 0, 0),
            False,
        ),
    ],
)
def test__is_parent_action__returns_expected(env, board, action, expected):
    board = torch.tensor(board, dtype=torch.uint8)
    assert env._is_parent_action(board, action) == expected


@pytest.mark.parametrize(
    "board, action, expected",
    [
        (
            [
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [4, 4, 4, 4],
                [4, 4, 4, 4],
            ],
            (4, 0, 0),
            False,
        ),
        (
            [
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [4, 4, 4, 4],
                [4, 4, 4, 4],
            ],
            (4, 0, 1),
            False,
        ),
        (
            [
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [4, 4, 4, 4],
                [4, 4, 4, 4],
            ],
            (4, 0, 2),
            True,
        ),
        (
            [
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [4, 4, 4, 4],
                [4, 4, 4, 4],
            ],
            (1, 0, 0),
            True,
        ),
    ],
)
def test__is_parent_action__returns_expected(env6x4, board, action, expected):
    board = torch.tensor(board, dtype=torch.uint8)
    _, is_parent = env6x4._is_parent_action(board, action)
    assert is_parent == expected


@pytest.mark.parametrize(
    "state, parents_expected, parents_a_expected",
    [
        (
            [
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [4, 4, 4, 4],
                [4, 4, 4, 4],
            ],
            [
                [
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [4, 4, 4, 4],
                    [4, 4, 4, 4],
                ],
                [
                    [1, 0, 0, 0],
                    [1, 0, 0, 0],
                    [1, 0, 0, 0],
                    [1, 0, 0, 0],
                    [4, 4, 0, 0],
                    [4, 4, 0, 0],
                ],
            ],
            [(1, 0, 0), (4, 0, 2)]
        ),
    ],
)
def test__get_parents__returns_expected(env6x4, state, parents_expected, parents_a_expected):
    state = torch.tensor(state, dtype=torch.uint8)
    parents_expected = [torch.tensor(parent, dtype=torch.uint8) for parent in parents_expected] 
    parents, parents_a = env6x4.get_parents(state)
    for p, p_e in zip(parents, parents_expected):
        assert torch.equal(p, p_e)
    for p_a, p_a_e in zip(parents_a, parents_a_expected):
        assert torch.equal(p_a, p_a_e)


# def test__all_env_common(env):
#     return common.test__all_env_common(env)
