from collections import Counter

import pytest

from gflownet.envs.grid import Grid
from gflownet.envs.tetris import Tetris


@pytest.fixture
def grid():
    return Grid()


@pytest.mark.parametrize(
    "state_x, state_y, is_equal",
    [
        ### Integers
        (
            0,
            0,
            True,
        ),
        (
            17,
            17,
            True,
        ),
        (
            17,
            18,
            False,
        ),
        ### Floats
        (
            0.0,
            0.0,
            True,
        ),
        (
            17.8,
            17.8,
            True,
        ),
        (
            17.0,
            18.0,
            False,
        ),
        (
            17.0,
            18.0,
            False,
        ),
        ### Lists
        (
            [],
            [],
            True,
        ),
        (
            [],
            [0],
            False,
        ),
        (
            [0],
            [0],
            True,
        ),
        (
            [0],
            [1],
            False,
        ),
        (
            [0, 1, -1],
            [0, 1, -1],
            True,
        ),
        (
            [0, 1, 1],
            [0, 1, -1],
            False,
        ),
        (
            [0, 1],
            [0, 1, -1],
            False,
        ),
        (
            [0.0, 1.0, -1.0],
            [0.0, 1.0, -1.0],
            True,
        ),
        (
            [0.0, 1.0, -1.0],
            [0.0, 1.0, 1.0],
            False,
        ),
        (
            ["a", "b", -1, 1],
            ["a", "b", -1, 1],
            True,
        ),
        (
            ["a", "b", -1, 0],
            ["a", "b", -1, 1],
            False,
        ),
        ### Lists of lists
        (
            [[0, 1], ["a", "b", -1, 0]],
            [[0, 1], ["a", "b", -1, 0]],
            True,
        ),
        (
            [[0, 1], ["a", "b", -1, 1]],
            [[0, 1], ["a", "b", -1, 0]],
            False,
        ),
        (
            [[0, 1], ["a", "b", -1, 0], 0.5],
            [[0, 1], ["a", "b", -1, 0], 0.5],
            True,
        ),
        (
            [[0, 1], ["a", "b", -1, 0], 0.5],
            [[0, 1], ["a", "b", -1, 0], 1.5],
            False,
        ),
        ### Dictionaries
        (
            {0: [1, 2, 3], 1: ["a", "b"]},
            {0: [1, 2, 3], 1: ["a", "b"]},
            True,
        ),
        # Key is different
        (
            {0: [1, 2, 3], 1: ["a", "b"]},
            {0: [1, 2, 3], 2: ["a", "b"]},
            False,
        ),
        # Value is different
        (
            {0: [1, 2, 3], 1: ["a", "b"]},
            {0: [1, 2, 3], 1: ["a", "c"]},
            False,
        ),
        # Order of keys are different
        (
            {0: [1, 2, 3], 1: ["a", "b"]},
            {1: ["a", "b"], 0: [1, 2, 3]},
            True,
        ),
        ### Counters
        (
            Counter(),
            Counter(),
            True,
        ),
        (
            Counter({1: 1}),
            Counter({1: 1}),
            True,
        ),
        (
            Counter({1: 1}),
            Counter({1: 2}),
            False,
        ),
        (
            Counter({1: 1}),
            Counter({2: 1}),
            False,
        ),
        (
            Counter({1: 1}),
            Counter(),
            False,
        ),
        ### Tuples
        (
            (),
            (),
            True,
        ),
        (
            (1,),
            (1,),
            True,
        ),
        (
            (1,),
            (2,),
            False,
        ),
        (
            (1, 2),
            (1, 2, 3),
            False,
        ),
        (
            (1, [0, 1], "a", (2, 3)),
            (1, [0, 1], "a", (2, 3)),
            True,
        ),
        (
            (1, [0, 1], "a", (2, 3)),
            (1, [0, 1], "b", (2, 3)),
            False,
        ),
        (
            (1, [0, 1], "a", (2, 3)),
            (1, [0, 1], "a", (2, 3, 1)),
            False,
        ),
        ### List of Counter and tuple (Wyckomposition)
        (
            [Counter(), ()],
            [Counter(), ()],
            True,
        ),
        (
            [Counter({(1, 1): 2}), ()],
            [Counter({(1, 1): 2}), ()],
            True,
        ),
        (
            [Counter({(1, 1): 2}), ()],
            [Counter({(1, 1): 3}), ()],
            False,
        ),
        (
            [Counter({(1, 1): 2}), ()],
            [Counter({(1, 1): 2}), (1, 2)],
            False,
        ),
        (
            [Counter({(1, 1): 2}), ()],
            [Counter(), ()],
            False,
        ),
    ],
)
def test__equal__behaves_as_expected(grid, state_x, state_y, is_equal):
    # The grid is use as a generic environment. Note that the values compared are not
    # grid states, but it does not matter for the purposes of this test.
    env = grid
    assert env.equal(state_x, state_y) == is_equal
