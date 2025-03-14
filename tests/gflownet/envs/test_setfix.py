import warnings

import common
import numpy as np
import pytest
import torch
from torch import Tensor

from gflownet.envs.cube import ContinuousCube
from gflownet.envs.grid import Grid
from gflownet.envs.set import SetFix
from gflownet.envs.stack import Stack
from gflownet.envs.tetris import Tetris
from gflownet.utils.common import copy, tbool


@pytest.fixture
def env_grid2d_tetrismini():
    return SetFix(
        subenvs=(
            Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),
            Tetris(
                width=4,
                height=5,
                pieces=["I", "O"],
                rotations=[0],
                allow_eos_before_full=True,
                device="cpu",
            ),
        )
    )


@pytest.fixture
def env_cube_tetris():
    return SetFix(
        subenvs=(
            ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
            Tetris(
                width=2,
                height=6,
                pieces=["J", "L", "O"],
                rotations=[0, 180],
                allow_eos_before_full=True,
                device="cpu",
            ),
        )
    )


@pytest.fixture
def env_cube_tetris_grid():
    return SetFix(
        subenvs=(
            ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
            Tetris(
                width=4,
                height=5,
                pieces=["I", "O"],
                rotations=[0],
                allow_eos_before_full=True,
                device="cpu",
            ),
            Grid(n_dim=3, length=3, cell_min=-1.0, cell_max=1.0),
        )
    )


@pytest.fixture
def env_two_grids():
    return SetFix(
        subenvs=(
            Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),
            Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),
        )
    )


@pytest.fixture
def env_three_cubes():
    return SetFix(
        subenvs=(
            ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
            ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
            ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
        )
    )


@pytest.fixture
def env_cube2d_cube3d():
    return SetFix(
        subenvs=(
            ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
            ContinuousCube(n_dim=3, n_comp=3, min_incr=0.1),
        )
    )


@pytest.fixture
def env_two_cubes2d_one_cube3d():
    return SetFix(
        subenvs=(
            ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
            ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
            ContinuousCube(n_dim=3, n_comp=3, min_incr=0.1),
        )
    )


@pytest.fixture
def env_stacks_equal():
    return SetFix(
        subenvs=(
            Stack(
                subenvs=(
                    ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
                    Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),
                )
            ),
            Stack(
                subenvs=(
                    ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
                    Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),
                )
            ),
        )
    )


@pytest.fixture
def env_stacks_diff():
    return SetFix(
        subenvs=(
            Stack(
                subenvs=(
                    ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
                    Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),
                )
            ),
            Stack(
                subenvs=(
                    Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),
                    ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
                )
            ),
        )
    )


@pytest.mark.parametrize(
    "env",
    [
        "env_grid2d_tetrismini",
        "env_cube_tetris",
        "env_cube_tetris_grid",
        "env_two_grids",
        "env_three_cubes",
        "env_cube2d_cube3d",
        "env_two_cubes2d_one_cube3d",
        "env_stacks_equal",
        "env_stacks_diff",
    ],
)
def test__environment__initializes_properly(env, request):
    env = request.getfixturevalue(env)
    assert True


@pytest.mark.parametrize(
    "env, n_unique_envs",
    [
        ("env_grid2d_tetrismini", 2),
        ("env_cube_tetris", 2),
        ("env_cube_tetris_grid", 3),
        ("env_two_grids", 1),
        ("env_three_cubes", 1),
        ("env_cube2d_cube3d", 2),
        ("env_two_cubes2d_one_cube3d", 2),
        ("env_stacks_equal", 1),
        ("env_stacks_diff", 2),
    ],
)
def test__number_of_unique_envs_is_correct(env, request, n_unique_envs):
    env = request.getfixturevalue(env)
    assert env.n_unique_envs == n_unique_envs


@pytest.mark.parametrize(
    "env, is_continuous",
    [
        ("env_grid2d_tetrismini", False),
        ("env_cube_tetris", True),
        ("env_cube_tetris_grid", True),
        ("env_two_grids", False),
        ("env_three_cubes", True),
        ("env_cube2d_cube3d", True),
        ("env_two_cubes2d_one_cube3d", True),
    ],
)
def test__environment__is_continuous(env, is_continuous, request):
    env = request.getfixturevalue(env)
    assert env.continuous == is_continuous


@pytest.mark.parametrize(
    "env, idx_subenv, idx_unique",
    [
        ("env_grid2d_tetrismini", 0, 0),
        ("env_grid2d_tetrismini", 1, 1),
        ("env_cube_tetris", 0, 0),
        ("env_cube_tetris", 1, 1),
        ("env_cube_tetris_grid", 0, 0),
        ("env_cube_tetris_grid", 1, 1),
        ("env_cube_tetris_grid", 2, 2),
        ("env_two_grids", 0, 0),
        ("env_two_grids", 1, 0),
        ("env_three_cubes", 0, 0),
        ("env_three_cubes", 1, 0),
        ("env_three_cubes", 2, 0),
        ("env_cube2d_cube3d", 0, 0),
        ("env_cube2d_cube3d", 1, 1),
        ("env_two_cubes2d_one_cube3d", 0, 0),
        ("env_two_cubes2d_one_cube3d", 1, 0),
        ("env_two_cubes2d_one_cube3d", 2, 1),
    ],
)
def test__get_unique_idx_of_subenv__returns_expected(
    env, request, idx_subenv, idx_unique
):
    env = request.getfixturevalue(env)
    assert env._get_unique_idx_of_subenv(idx_subenv) == idx_unique


@pytest.mark.parametrize(
    "env, source",
    [
        (
            "env_grid2d_tetrismini",
            [
                [-1, 0, [0, 0], [0, 1]],
                {
                    0: [0, 0],
                    1: torch.tensor(
                        [
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                        ],
                        dtype=torch.int16,
                    ),
                },
            ],
        ),
        (
            "env_cube_tetris",
            [
                [-1, 0, [0, 0], [0, 1]],
                {
                    0: [-1, -1],
                    1: torch.tensor(
                        [
                            [000, 000],
                            [000, 000],
                            [000, 000],
                            [000, 000],
                            [000, 000],
                            [000, 000],
                        ],
                        dtype=torch.int16,
                    ),
                },
            ],
        ),
        (
            "env_cube_tetris_grid",
            [
                [-1, 0, [0, 0, 0], [0, 1, 2]],
                {
                    0: [-1, -1],
                    1: torch.tensor(
                        [
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                        ],
                        dtype=torch.int16,
                    ),
                    2: [0, 0, 0],
                },
            ],
        ),
        ("env_two_grids", [[-1, 0, [0, 0], [0, 0]], {0: [0, 0], 1: [0, 0]}]),
        (
            "env_three_cubes",
            [[-1, 0, [0, 0, 0], [0, 0, 0]], {0: [-1, -1], 1: [-1, -1], 2: [-1, -1]}],
        ),
        (
            "env_two_cubes2d_one_cube3d",
            [
                [-1, 0, [0, 0, 0], [0, 0, 1]],
                {0: [-1, -1], 1: [-1, -1], 2: [-1, -1, -1]},
            ],
        ),
    ],
)
def test__source_state_is_expected(env, source, request):
    env = request.getfixturevalue(env)
    assert env.equal(env.source, source)


@pytest.mark.parametrize(
    "env, state, is_source",
    [
        (
            "env_grid2d_tetrismini",
            [
                [-1, 0, [0, 0], [0, 1]],
                {
                    0: [0, 0],
                    1: torch.tensor(
                        [
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                        ],
                        dtype=torch.int16,
                    ),
                },
            ],
            True,
        ),
        (
            "env_grid2d_tetrismini",
            [
                [0, 1, [0, 0], [0, 1]],
                {
                    0: [0, 0],
                    1: torch.tensor(
                        [
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                        ],
                        dtype=torch.int16,
                    ),
                },
            ],
            False,
        ),
        (
            "env_grid2d_tetrismini",
            [
                [-1, 0, [1, 0], [0, 1]],
                {
                    0: [0, 0],
                    1: torch.tensor(
                        [
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                        ],
                        dtype=torch.int16,
                    ),
                },
            ],
            False,
        ),
        (
            "env_cube_tetris",
            [
                [-1, 0, [0, 0], [0, 1]],
                {
                    0: [-1, -1],
                    1: torch.tensor(
                        [
                            [000, 000],
                            [000, 000],
                            [000, 000],
                            [000, 000],
                            [000, 000],
                            [000, 000],
                        ],
                        dtype=torch.int16,
                    ),
                },
            ],
            True,
        ),
        (
            "env_cube_tetris",
            [
                [-1, 0, [0, 0], [1, 1]],
                {
                    0: [-1, -1],
                    1: torch.tensor(
                        [
                            [000, 000],
                            [000, 000],
                            [000, 000],
                            [000, 000],
                            [000, 000],
                            [000, 000],
                        ],
                        dtype=torch.int16,
                    ),
                },
            ],
            False,
        ),
        (
            "env_cube_tetris_grid",
            [
                [-1, 0, [0, 0, 0], [0, 1, 2]],
                {
                    0: [-1, -1],
                    1: torch.tensor(
                        [
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                        ],
                        dtype=torch.int16,
                    ),
                    2: [0, 0, 0],
                },
            ],
            True,
        ),
        (
            "env_cube_tetris_grid",
            [
                [-1, 1, [0, 0, 0], [0, 1, 2]],
                {
                    0: [-1, -1],
                    1: torch.tensor(
                        [
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                        ],
                        dtype=torch.int16,
                    ),
                    2: [0, 0, 0],
                },
            ],
            False,
        ),
        ("env_two_grids", [[-1, 0, [0, 0], [0, 0]], {0: [0, 0], 1: [0, 0]}], True),
        ("env_two_grids", [[-1, 0, [0, 0], [0, 0]], {0: [0, 0], 1: [0, 1]}], False),
        ("env_two_grids", [[-1, 0, [0, 0], [0, 0]], {0: [1, 0], 1: [0, 0]}], False),
        (
            "env_three_cubes",
            [[-1, 0, [0, 0, 0], [0, 0, 0]], {0: [-1, -1], 1: [-1, -1], 2: [-1, -1]}],
            True,
        ),
        (
            "env_three_cubes",
            [[-1, 0, [0, 0, 1], [0, 0, 0]], {0: [-1, -1], 1: [-1, -1], 2: [-1, -1]}],
            False,
        ),
        (
            "env_two_cubes2d_one_cube3d",
            [
                [-1, 0, [0, 0, 0], [0, 0, 1]],
                {0: [-1, -1], 1: [-1, -1], 2: [-1, -1, -1]},
            ],
            True,
        ),
        (
            "env_two_cubes2d_one_cube3d",
            [
                [0, 0, [0, 0, 0], [0, 0, 1]],
                {0: [-1, -1], 1: [-1, -1], 2: [-1, -1, -1]},
            ],
            False,
        ),
        (
            "env_stacks_equal",
            [
                # fmt: off
                [-1, 0, [0, 0], [0, 0]],
                {
                    0: [0, [-1, -1], [0, 0]],
                    1: [0, [-1, -1], [0, 0]],
                },
                # fmt: on
            ],
            True,
        ),
        (
            "env_stacks_equal",
            [
                # fmt: off
                [-1, 0, [0, 0], [0, 0]],
                {
                    0: [1, [-1, -1], [0, 0]],
                    1: [0, [-1, -1], [0, 0]],
                },
                # fmt: on
            ],
            False,
        ),
    ],
)
def test__is_source__returns_expected(env, state, is_source, request):
    env = request.getfixturevalue(env)
    assert env.is_source(state) == is_source
    env.set_state(state)
    assert env.is_source() == is_source


@pytest.mark.parametrize(
    "env, action_space",
    [
        (
            "env_grid2d_tetrismini",
            [
                # fmt: off
                # Activate subenvs
                (-1, 0, 0, 0),
                (-1, 1, 0, 0),
                # EOS
                (-1, -1, -1, -1),
                # Grid
                (0, 0, 0, 0),
                (0, 1, 0, 0),
                (0, 0, 1, 0),
                # Tetris
                (1, 1, 0, 0),
                (1, 1, 0, 1),
                (1, 1, 0, 2),
                (1, 1, 0, 3),
                (1, 4, 0, 0),
                (1, 4, 0, 1),
                (1, 4, 0, 2),
                (1, -1, -1, -1),
                # fmt: on
            ],
        ),
        (
            "env_two_grids",
            [
                # fmt: off
                # Activate subenvs
                (-1, 0, 0),
                (-1, 1, 0),
                # EOS
                (-1, -1, -1),
                # Grid
                (0, 0, 0),
                (0, 1, 0),
                (0, 0, 1),
                # fmt: on
            ],
        ),
    ],
)
def test__get_action_space__returns_expected(env, action_space, request):
    env = request.getfixturevalue(env)
    assert set(action_space) == set(env.action_space)


@pytest.mark.parametrize(
    "env, mask_dim_expected",
    [
        ("env_grid2d_tetrismini", max([3, 8, 2 + 1]) + 2),
        ("env_cube_tetris", max([5, 6, 2 + 1]) + 2),
        ("env_cube_tetris_grid", max([5, 8, 3, 3 + 1]) + 3),
        ("env_two_grids", max([3, 3, 2 + 1]) + 2),
        ("env_three_cubes", max([5, 5, 5, 3 + 1]) + 3),
        ("env_cube2d_cube3d", max([5, 6, 2 + 1]) + 2),
        ("env_two_cubes2d_one_cube3d", max([5, 5, 6, 3 + 1]) + 3),
    ],
)
def test__mask_dim__is_as_expected(env, request, mask_dim_expected):
    env = request.getfixturevalue(env)
    assert env.mask_dim == mask_dim_expected


@pytest.mark.parametrize(
    "env, action_set, action_subenv",
    [
        ("env_grid2d_tetrismini", (0, 0, 0, 0), (0, 0)),
        ("env_grid2d_tetrismini", (0, 1, 0, 0), (1, 0)),
        ("env_grid2d_tetrismini", (0, 0, 1, 0), (0, 1)),
        ("env_grid2d_tetrismini", (1, 1, 0, 0), (1, 0, 0)),
        ("env_grid2d_tetrismini", (1, 1, 0, 3), (1, 0, 3)),
        ("env_grid2d_tetrismini", (1, 4, 0, 2), (4, 0, 2)),
        ("env_cube_tetris", (0, 0.2, 0.3, 0), (0.2, 0.3, 0)),
        ("env_cube_tetris", (0, 0.5, 0.7, 1), (0.5, 0.7, 1)),
        ("env_cube_tetris", (0, np.inf, np.inf, np.inf), (np.inf, np.inf, np.inf)),
        ("env_cube_tetris", (1, 1, 0, 3), (1, 0, 3)),
        ("env_cube_tetris", (1, 4, 0, 2), (4, 0, 2)),
        ("env_cube_tetris_grid", (0, 0.5, 0.7, 1), (0.5, 0.7, 1)),
        ("env_cube_tetris_grid", (1, 4, 0, 2), (4, 0, 2)),
        ("env_cube_tetris_grid", (2, 0, 1, 0), (0, 1, 0)),
        ("env_cube_tetris_grid", (2, 0, 0, 0), (0, 0, 0)),
        ("env_two_grids", (0, 0, 0), (0, 0)),
        ("env_two_grids", (0, 1, 0), (1, 0)),
        ("env_two_grids", (0, 0, 1), (0, 1)),
        ("env_two_grids", (-1, 0, 0), (0,)),
        ("env_two_grids", (-1, 1, 0), (1,)),
        ("env_two_cubes2d_one_cube3d", (0, 0.2, 0.3, 0, 0), (0.2, 0.3, 0)),
        ("env_two_cubes2d_one_cube3d", (0, 0.2, 0.3, 1, 0), (0.2, 0.3, 1)),
        ("env_two_cubes2d_one_cube3d", (1, 0.5, 0.7, 0.2, 0), (0.5, 0.7, 0.2, 0)),
        ("env_two_cubes2d_one_cube3d", (1, 0.5, 0.7, 0.2, 1), (0.5, 0.7, 0.2, 1)),
        ("env_two_cubes2d_one_cube3d", (-1, 0, 0, 0, 0), (0,)),
        ("env_two_cubes2d_one_cube3d", (-1, 1, 0, 0, 0), (1,)),
    ],
)
def test__pad_depad_action__return_expected(env, action_set, action_subenv, request):
    env = request.getfixturevalue(env)
    idx_env_unique = action_set[0]
    # Check pad
    assert env._pad_action(action_subenv, idx_env_unique) == action_set
    # Check depad
    assert env._depad_action(action_set, idx_env_unique) == action_subenv


@pytest.mark.parametrize(
    "env, state, done",
    [
        (
            "env_grid2d_tetrismini",
            [
                [-1, 0, [0, 0], [0, 1]],
                {
                    0: [0, 0],
                    1: torch.tensor(
                        [
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                        ],
                        dtype=torch.int16,
                    ),
                },
            ],
            False,
        ),
        (
            "env_grid2d_tetrismini",
            [
                [0, 1, [0, 0], [0, 1]],
                {
                    0: [1, 2],
                    1: torch.tensor(
                        [
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [400, 400, 000, 000],
                            [400, 400, 000, 000],
                        ],
                        dtype=torch.int16,
                    ),
                },
            ],
            False,
        ),
        (
            "env_grid2d_tetrismini",
            [
                [0, 0, [1, 1], [0, 1]],
                {
                    0: [1, 2],
                    1: torch.tensor(
                        [
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [400, 400, 000, 000],
                            [400, 400, 000, 000],
                        ],
                        dtype=torch.int16,
                    ),
                },
            ],
            True,
        ),
        (
            "env_grid2d_tetrismini",
            [
                [0, 1, [0, 1], [0, 1]],
                {
                    0: [1, 2],
                    1: torch.tensor(
                        [
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [400, 400, 000, 000],
                            [400, 400, 000, 000],
                        ],
                        dtype=torch.int16,
                    ),
                },
            ],
            False,
        ),
        (
            "env_grid2d_tetrismini",
            [
                [0, 0, [1, 1], [0, 1]],
                {
                    0: [1, 2],
                    1: torch.tensor(
                        [
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [400, 400, 000, 000],
                            [400, 400, 000, 000],
                        ],
                        dtype=torch.int16,
                    ),
                },
            ],
            True,
        ),
    ],
)
def test__set_state__sets_state_and_dones(env, state, done, request):
    env = request.getfixturevalue(env)
    env.set_state(state, done)

    # Check global state
    assert env.equal(env.state, state)

    # Check states of subenvs
    for idx, subenv in enumerate(env.subenvs):
        assert env.equal(subenv.state, env._get_substate(state, idx))

    # Check parent done
    assert env.done == done

    # Check dones
    dones = env._get_dones()
    for subenv, done in zip(env.subenvs, dones):
        assert subenv.done == done


@pytest.mark.parametrize(
    "env, state_from, action, state_next_exp, valid_exp",
    [
        # From source: activate grid
        (
            "env_grid2d_tetrismini",
            [
                [-1, 0, [0, 0], [0, 1]],
                {
                    0: [0, 0],
                    1: torch.tensor(
                        [
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                        ],
                        dtype=torch.int16,
                    ),
                },
            ],
            (-1, 0, 0, 0),
            [
                [0, 1, [0, 0], [0, 1]],
                {
                    0: [0, 0],
                    1: torch.tensor(
                        [
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                        ],
                        dtype=torch.int16,
                    ),
                },
            ],
            True,
        ),
        # From source -> activate grid: grid action
        (
            "env_grid2d_tetrismini",
            [
                [0, 1, [0, 0], [0, 1]],
                {
                    0: [0, 0],
                    1: torch.tensor(
                        [
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                        ],
                        dtype=torch.int16,
                    ),
                },
            ],
            (0, 1, 0, 0),
            [
                [0, 0, [0, 0], [0, 1]],
                {
                    0: [1, 0],
                    1: torch.tensor(
                        [
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                        ],
                        dtype=torch.int16,
                    ),
                },
            ],
            True,
        ),
        # From source -> activate grid -> grid action: deactivate grid
        (
            "env_grid2d_tetrismini",
            [
                [0, 0, [0, 0], [0, 1]],
                {
                    0: [1, 0],
                    1: torch.tensor(
                        [
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                        ],
                        dtype=torch.int16,
                    ),
                },
            ],
            (-1, 0, 0, 0),
            [
                [-1, 0, [0, 0], [0, 1]],
                {
                    0: [1, 0],
                    1: torch.tensor(
                        [
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                        ],
                        dtype=torch.int16,
                    ),
                },
            ],
            True,
        ),
        # From intermediate grid (but active -1): activate tetris
        (
            "env_grid2d_tetrismini",
            [
                [-1, 0, [0, 0], [0, 1]],
                {
                    0: [1, 1],
                    1: torch.tensor(
                        [
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                        ],
                        dtype=torch.int16,
                    ),
                },
            ],
            (-1, 1, 0, 0),
            [
                [1, 1, [0, 0], [0, 1]],
                {
                    0: [1, 1],
                    1: torch.tensor(
                        [
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                        ],
                        dtype=torch.int16,
                    ),
                },
            ],
            True,
        ),
        # From source: activate 0th grid
        (
            "env_two_grids",
            [
                [-1, 0, [0, 0], [0, 0]],
                {
                    0: [0, 0],
                    1: [0, 0],
                },
            ],
            (-1, 0, 0),
            [
                [0, 1, [0, 0], [0, 0]],
                {
                    0: [0, 0],
                    1: [0, 0],
                },
            ],
            True,
        ),
        # From source: activate 1st grid
        (
            "env_two_grids",
            [
                [-1, 0, [0, 0], [0, 0]],
                {
                    0: [0, 0],
                    1: [0, 0],
                },
            ],
            (-1, 1, 0),
            [
                [1, 1, [0, 0], [0, 0]],
                {
                    0: [0, 0],
                    1: [0, 0],
                },
            ],
            True,
        ),
        # From source -> activate 0th grid: grid action
        (
            "env_two_grids",
            [
                [0, 1, [0, 0], [0, 0]],
                {
                    0: [0, 0],
                    1: [0, 0],
                },
            ],
            (0, 1, 0),
            [
                [0, 0, [0, 0], [0, 0]],
                {
                    0: [1, 0],
                    1: [0, 0],
                },
            ],
            True,
        ),
        # From active 0th grid (toggle flag 0): deactivate 0th grid
        (
            "env_two_grids",
            [
                [0, 0, [0, 0], [0, 0]],
                {
                    0: [1, 0],
                    1: [0, 0],
                },
            ],
            (-1, 0, 0),
            [
                [-1, 0, [0, 0], [0, 0]],
                {
                    0: [1, 0],
                    1: [0, 0],
                },
            ],
            True,
        ),
        # From source -> activate 1th grid: grid action
        (
            "env_two_grids",
            [
                [1, 1, [0, 0], [0, 0]],
                {
                    0: [0, 0],
                    1: [0, 0],
                },
            ],
            (0, 1, 0),
            [
                [1, 0, [0, 0], [0, 0]],
                {
                    0: [0, 0],
                    1: [1, 0],
                },
            ],
            True,
        ),
        # From intermediate grid (but active -1): activate the other grid
        (
            "env_two_grids",
            [
                [-1, 0, [0, 0], [0, 0]],
                {
                    0: [0, 0],
                    1: [1, 0],
                },
            ],
            (-1, 0, 0),
            [
                [0, 1, [0, 0], [0, 0]],
                {
                    0: [0, 0],
                    1: [1, 0],
                },
            ],
            True,
        ),
        # From source -> activate Cube 3D: Cube 3D action
        (
            "env_two_cubes2d_one_cube3d",
            [
                [2, 1, [0, 0, 0], [0, 0, 1]],
                {
                    0: [-1, -1],
                    1: [-1, -1],
                    2: [-1, -1, -1],
                },
            ],
            (1, 0.3, 0.2, 0.7, 1),
            [
                [2, 0, [0, 0, 0], [0, 0, 1]],
                {
                    0: [-1, -1],
                    1: [-1, -1],
                    2: [0.3, 0.2, 0.7],
                },
            ],
            True,
        ),
        # From intermediate Cube 3D (but active -1): activate 0th Cube 2D
        (
            "env_two_cubes2d_one_cube3d",
            [
                [-1, 0, [0, 0, 0], [0, 0, 1]],
                {
                    0: [-1, -1],
                    1: [-1, -1],
                    2: [0.3, 0.2, 0.7],
                },
            ],
            (-1, 0, 0, 0, 0),
            [
                [0, 1, [0, 0, 0], [0, 0, 1]],
                {
                    0: [-1, -1],
                    1: [-1, -1],
                    2: [0.3, 0.2, 0.7],
                },
            ],
            True,
        ),
        # From intermediate 0th Cube 2D (but active -1), activate 1st Cube 2D
        (
            "env_two_cubes2d_one_cube3d",
            [
                [-1, 0, [0, 0, 0], [0, 0, 1]],
                {
                    0: [0.4, 0.6],
                    1: [-1, -1],
                    2: [0.3, 0.2, 0.7],
                },
            ],
            (-1, 1, 0, 0, 0),
            [
                [1, 1, [0, 0, 0], [0, 0, 1]],
                {
                    0: [0.4, 0.6],
                    1: [-1, -1],
                    2: [0.3, 0.2, 0.7],
                },
            ],
            True,
        ),
    ],
)
def test__step__works_as_expected(
    env, state_from, action, state_next_exp, valid_exp, request
):
    env = request.getfixturevalue(env)
    env.set_state(state_from)

    # Check init state
    assert env.equal(env.state, state_from)

    # Perform step
    state_next, action_done, valid = env.step(action, skip_mask_check=True)

    # Check end state
    assert env.equal(env.state, state_next)
    assert env.equal(env.state, state_next_exp)

    # Check action and valid
    assert action_done == action
    assert valid == valid_exp, (state_from, action)

    # Check done
    if action == env.eos and valid_exp:
        assert env.done


@pytest.mark.parametrize(
    "env, state_from, action, state_next_exp, valid_exp",
    [
        # From source -> activate grid: toggle (deactivate) grid
        (
            "env_grid2d_tetrismini",
            [
                [0, 1, [0, 0], [0, 1]],
                {
                    0: [0, 0],
                    1: torch.tensor(
                        [
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                        ],
                        dtype=torch.int16,
                    ),
                },
            ],
            (-1, 0, 0, 0),
            [
                [-1, 0, [0, 0], [0, 1]],
                {
                    0: [0, 0],
                    1: torch.tensor(
                        [
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                        ],
                        dtype=torch.int16,
                    ),
                },
            ],
            True,
        ),
        # From intermediate grid (toggle flag 0) -> grid action:
        (
            "env_grid2d_tetrismini",
            [
                [0, 0, [0, 0], [0, 1]],
                {
                    0: [1, 0],
                    1: torch.tensor(
                        [
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                        ],
                        dtype=torch.int16,
                    ),
                },
            ],
            (0, 1, 0, 0),
            [
                [0, 1, [0, 0], [0, 1]],
                {
                    0: [0, 0],
                    1: torch.tensor(
                        [
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                        ],
                        dtype=torch.int16,
                    ),
                },
            ],
            True,
        ),
        # From intermediate state (inactive): toggle (activate) grid
        (
            "env_grid2d_tetrismini",
            [
                [-1, 0, [0, 0], [0, 1]],
                {
                    0: [1, 0],
                    1: torch.tensor(
                        [
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                        ],
                        dtype=torch.int16,
                    ),
                },
            ],
            (-1, 0, 0, 0),
            [
                [0, 0, [0, 0], [0, 1]],
                {
                    0: [1, 0],
                    1: torch.tensor(
                        [
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                        ],
                        dtype=torch.int16,
                    ),
                },
            ],
            True,
        ),
        # From active tetris (toggle flag 1): toggle (deactivate) tetris
        (
            "env_grid2d_tetrismini",
            [
                [1, 1, [0, 0], [0, 1]],
                {
                    0: [1, 1],
                    1: torch.tensor(
                        [
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                        ],
                        dtype=torch.int16,
                    ),
                },
            ],
            (-1, 1, 0, 0),
            [
                [-1, 0, [0, 0], [0, 1]],
                {
                    0: [1, 1],
                    1: torch.tensor(
                        [
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                        ],
                        dtype=torch.int16,
                    ),
                },
            ],
            True,
        ),
        # From active 0th grid: toggle (deactivate) 0th grid
        (
            "env_two_grids",
            [
                [0, 1, [0, 0], [0, 0]],
                {
                    0: [0, 0],
                    1: [0, 0],
                },
            ],
            (-1, 0, 0),
            [
                [-1, 0, [0, 0], [0, 0]],
                {
                    0: [0, 0],
                    1: [0, 0],
                },
            ],
            True,
        ),
        # From active 1st grid: toggle (deactivate) 1st grid
        (
            "env_two_grids",
            [
                [1, 1, [0, 0], [0, 0]],
                {
                    0: [0, 0],
                    1: [0, 0],
                },
            ],
            (-1, 1, 0),
            [
                [-1, 0, [0, 0], [0, 0]],
                {
                    0: [0, 0],
                    1: [0, 0],
                },
            ],
            True,
        ),
        # From active 0th grid (toggle flag 0): grid action
        (
            "env_two_grids",
            [
                [0, 0, [0, 0], [0, 0]],
                {
                    0: [1, 0],
                    1: [0, 0],
                },
            ],
            (0, 1, 0),
            [
                [0, 1, [0, 0], [0, 0]],
                {
                    0: [0, 0],
                    1: [0, 0],
                },
            ],
            True,
        ),
        # From intermediate state (inactive): toggle (activate) grid 0th
        (
            "env_two_grids",
            [
                [-1, 0, [0, 0], [0, 0]],
                {
                    0: [1, 0],
                    1: [0, 0],
                },
            ],
            (-1, 0, 0),
            [
                [0, 0, [0, 0], [0, 0]],
                {
                    0: [1, 0],
                    1: [0, 0],
                },
            ],
            True,
        ),
        # From active 1st grid (toggle flag 0): grid action
        (
            "env_two_grids",
            [
                [1, 0, [0, 0], [0, 0]],
                {
                    0: [0, 0],
                    1: [1, 0],
                },
            ],
            (0, 1, 0),
            [
                [1, 1, [0, 0], [0, 0]],
                {
                    0: [0, 0],
                    1: [0, 0],
                },
            ],
            True,
        ),
        # From active grid (toggle flag 1): toggle (deactivate) grid
        (
            "env_two_grids",
            [
                [0, 1, [0, 0], [0, 0]],
                {
                    0: [0, 0],
                    1: [1, 0],
                },
            ],
            (-1, 0, 0),
            [
                [-1, 0, [0, 0], [0, 0]],
                {
                    0: [0, 0],
                    1: [1, 0],
                },
            ],
            True,
        ),
        # From active Cube 3D (toggle flag 0): Cube 3D action
        (
            "env_two_cubes2d_one_cube3d",
            [
                [2, 0, [0, 0, 0], [0, 0, 1]],
                {
                    0: [-1, -1],
                    1: [-1, -1],
                    2: [0.3, 0.2, 0.7],
                },
            ],
            (1, 0.3, 0.2, 0.7, 1),
            [
                [2, 1, [0, 0, 0], [0, 0, 1]],
                {
                    0: [-1, -1],
                    1: [-1, -1],
                    2: [-1, -1, -1],
                },
            ],
            True,
        ),
        # From active 0th Cube 2D (toggle flag 1): toggle (deactivate) 0th Cube 2D
        (
            "env_two_cubes2d_one_cube3d",
            [
                [0, 1, [0, 0, 0], [0, 0, 1]],
                {
                    0: [-1, -1],
                    1: [-1, -1],
                    2: [0.3, 0.2, 0.7],
                },
            ],
            (-1, 0, 0, 0, 0),
            [
                [-1, 0, [0, 0, 0], [0, 0, 1]],
                {
                    0: [-1, -1],
                    1: [-1, -1],
                    2: [0.3, 0.2, 0.7],
                },
            ],
            True,
        ),
        # From active 1st Cube 2D (toggle flag 1): toggle (deactivate) 1st Cube 2D
        (
            "env_two_cubes2d_one_cube3d",
            [
                [1, 1, [0, 0, 0], [0, 0, 1]],
                {
                    0: [0.4, 0.6],
                    1: [-1, -1],
                    2: [0.3, 0.2, 0.7],
                },
            ],
            (-1, 1, 0, 0, 0),
            [
                [-1, 0, [0, 0, 0], [0, 0, 1]],
                {
                    0: [0.4, 0.6],
                    1: [-1, -1],
                    2: [0.3, 0.2, 0.7],
                },
            ],
            True,
        ),
    ],
)
def test__step_backwards__works_as_expected(
    env, state_from, action, state_next_exp, valid_exp, request
):
    env = request.getfixturevalue(env)
    env.set_state(state_from)

    # Check init state
    assert env.equal(env.state, state_from)

    # Perform step
    state_next, action_done, valid = env.step_backwards(action, skip_mask_check=True)

    # Check end state
    assert env.equal(env.state, state_next)
    assert env.equal(env.state, state_next_exp)

    # Check action and valid
    assert action_done == action
    assert valid == valid_exp, (state_from, action)


@pytest.mark.parametrize(
    "env, state_from",
    [
        (
            "env_grid2d_tetrismini",
            [
                [-1, 0, [1, 1], [0, 1]],
                {
                    0: [1, 1],
                    1: torch.tensor(
                        [
                            [000, 000, 000, 000],
                            [000, 000, 100, 000],
                            [000, 000, 100, 000],
                            [400, 400, 100, 000],
                            [400, 400, 100, 000],
                        ],
                        dtype=torch.int16,
                    ),
                },
            ],
        ),
        (
            "env_two_grids",
            [
                [-1, 0, [1, 1], [0, 0]],
                {
                    0: [1, 2],
                    1: [1, 0],
                },
            ],
        ),
        (
            "env_two_cubes2d_one_cube3d",
            [
                [-1, 0, [1, 1, 1], [0, 0, 1]],
                {
                    0: [0.4, 0.6],
                    1: [0.15, 0.67],
                    2: [0.3, 0.2, 0.7],
                },
            ],
        ),
    ],
)
def test__step__eos_action_valid_if_all_subenvs_are_done(env, state_from, request):
    env = request.getfixturevalue(env)
    # Set state with all subenvs done
    env.set_state(state_from, done=False)

    state_next, action_done, valid = env.step(env.eos, skip_mask_check=True)

    # Check end state
    assert env.equal(env.state, state_next)
    assert env.equal(env.state, state_from)

    # Check action and valid
    assert action_done == env.eos
    assert valid

    # Check done
    assert env.done


@pytest.mark.parametrize(
    "env, state_from",
    [
        (
            "env_grid2d_tetrismini",
            [
                [-1, 0, [1, 1], [0, 1]],
                {
                    0: [1, 1],
                    1: torch.tensor(
                        [
                            [000, 000, 000, 000],
                            [000, 000, 100, 000],
                            [000, 000, 100, 000],
                            [400, 400, 100, 000],
                            [400, 400, 100, 000],
                        ],
                        dtype=torch.int16,
                    ),
                },
            ],
        ),
        (
            "env_two_grids",
            [
                [-1, 0, [1, 1], [0, 0]],
                {
                    0: [1, 2],
                    1: [1, 0],
                },
            ],
        ),
        (
            "env_two_cubes2d_one_cube3d",
            [
                [-1, 0, [1, 1, 1], [0, 0, 1]],
                {
                    0: [0.4, 0.6],
                    1: [0.15, 0.67],
                    2: [0.3, 0.2, 0.7],
                },
            ],
        ),
    ],
)
def test__step_backwards__eos_action_valid_if_all_subenvs_are_done(
    env, state_from, request
):
    env = request.getfixturevalue(env)
    # Set state with all subenvs done
    env.set_state(state_from, done=True)

    state_next, action_done, valid = env.step_backwards(env.eos, skip_mask_check=True)

    # Check end state
    assert env.equal(env.state, state_next)
    assert env.equal(env.state, state_from)

    # Check action and valid
    assert action_done == env.eos
    assert valid

    # Check done
    assert not env.done


@pytest.mark.parametrize(
    "env, state, mask_exp",
    [
        # Source
        (
            "env_grid2d_tetrismini",
            [
                [-1, 0, [0, 0], [0, 1]],
                {
                    0: [0, 0],
                    1: torch.tensor(
                        [
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                        ],
                        dtype=torch.int16,
                    ),
                },
            ],
            # fmt: off
            [
                False, False, # ACTIVE SUBENV
                False, False, True, # MASK
                False, False, False, False, False # PAD
            ]
            # fmt: on
        ),
        # Source -> activate grid
        (
            "env_grid2d_tetrismini",
            [
                [0, 1, [0, 0], [0, 1]],
                {
                    0: [0, 0],
                    1: torch.tensor(
                        [
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                        ],
                        dtype=torch.int16,
                    ),
                },
            ],
            # fmt: off
            [
                True, False, # ACTIVE SUBENV
                False, False, False, # MASK
                False, False, False, False, False # PAD
            ]
            # fmt: on
        ),
        # Source -> activate tetris
        (
            "env_grid2d_tetrismini",
            [
                [1, 1, [0, 0], [0, 1]],
                {
                    0: [0, 0],
                    1: torch.tensor(
                        [
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                        ],
                        dtype=torch.int16,
                    ),
                },
            ],
            # fmt: off
            [
                False, True, # ACTIVE SUBENV
                False, False, False, False, False, False, False, False # MASK
            ]
            # fmt: on
        ),
        # Intermediate state, no active subenv
        (
            "env_grid2d_tetrismini",
            [
                [-1, 0, [0, 0], [0, 1]],
                {
                    0: [2, 1],
                    1: torch.tensor(
                        [
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [400, 400, 000, 000],
                            [400, 400, 000, 000],
                        ],
                        dtype=torch.int16,
                    ),
                },
            ],
            # fmt: off
            [
                False, False, # ACTIVE SUBENV
                False, False, True, # MASK
                False, False, False, False, False # PAD
            ]
            # fmt: on
        ),
        # Intermediate state, grid active (before subenv action)
        (
            "env_grid2d_tetrismini",
            [
                [0, 1, [0, 0], [0, 1]],
                {
                    0: [2, 1],
                    1: torch.tensor(
                        [
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [400, 400, 000, 000],
                            [400, 400, 000, 000],
                        ],
                        dtype=torch.int16,
                    ),
                },
            ],
            # fmt: off
            [
                True, False, # ACTIVE SUBENV
                False, True, False, # MASK
                False, False, False, False, False # PAD
            ]
            # fmt: on
        ),
        # Intermediate state, grid active (after subenv action)
        (
            "env_grid2d_tetrismini",
            [
                [0, 0, [0, 0], [0, 1]],
                {
                    0: [2, 1],
                    1: torch.tensor(
                        [
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [400, 400, 000, 000],
                            [400, 400, 000, 000],
                        ],
                        dtype=torch.int16,
                    ),
                },
            ],
            # fmt: off
            [
                False, False, # ACTIVE SUBENV
                False, True, True, # MASK
                False, False, False, False, False # PAD
            ]
            # fmt: on
        ),
    ],
)
def test__get_mask_invalid_actions_forward__returns_expected(
    env, state, mask_exp, request
):
    env = request.getfixturevalue(env)
    mask = env.get_mask_invalid_actions_forward(state, done=False)
    assert mask == mask_exp


@pytest.mark.parametrize(
    "env, state, mask_exp",
    [
        # Source
        (
            "env_grid2d_tetrismini",
            [
                [-1, 0, [0, 0], [0, 1]],
                {
                    0: [0, 0],
                    1: torch.tensor(
                        [
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                        ],
                        dtype=torch.int16,
                    ),
                },
            ],
            # fmt: off
            [
                False, False, # ACTIVE SUBENV
                True, True, True, # MASK
                False, False, False, False, False # PAD
            ]
            # fmt: on
        ),
        # Source -> activate grid
        (
            "env_grid2d_tetrismini",
            [
                [0, 1, [0, 0], [0, 1]],
                {
                    0: [0, 0],
                    1: torch.tensor(
                        [
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                        ],
                        dtype=torch.int16,
                    ),
                },
            ],
            # fmt: off
            [
                False, False, # ACTIVE SUBENV
                False, True, True, # MASK
                False, False, False, False, False # PAD
            ]
            # fmt: on
        ),
        # Source -> activate tetris
        (
            "env_grid2d_tetrismini",
            [
                [1, 1, [0, 0], [0, 1]],
                {
                    0: [0, 0],
                    1: torch.tensor(
                        [
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                        ],
                        dtype=torch.int16,
                    ),
                },
            ],
            # fmt: off
            [
                False, False, # ACTIVE SUBENV
                True, False, True, # MASK
                False, False, False, False, False # PAD
            ]
            # fmt: on
        ),
        # Intermediate state, no active subenv, both states no source
        (
            "env_grid2d_tetrismini",
            [
                [-1, 0, [0, 0], [0, 1]],
                {
                    0: [2, 1],
                    1: torch.tensor(
                        [
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [400, 400, 000, 000],
                            [400, 400, 000, 000],
                        ],
                        dtype=torch.int16,
                    ),
                },
            ],
            # fmt: off
            [
                False, False, # ACTIVE SUBENV
                False, False, True, # MASK
                False, False, False, False, False # PAD
            ]
            # fmt: on
        ),
        # Intermediate state, no active subenv, tetris is source
        (
            "env_grid2d_tetrismini",
            [
                [-1, 0, [0, 0], [0, 1]],
                {
                    0: [2, 1],
                    1: torch.tensor(
                        [
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                        ],
                        dtype=torch.int16,
                    ),
                },
            ],
            # fmt: off
            [
                False, False, # ACTIVE SUBENV
                False, True, True, # MASK
                False, False, False, False, False # PAD
            ]
            # fmt: on
        ),
        # Intermediate state, no active subenv, grid is source
        (
            "env_grid2d_tetrismini",
            [
                [-1, 0, [0, 0], [0, 1]],
                {
                    0: [0, 0],
                    1: torch.tensor(
                        [
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [400, 400, 000, 000],
                            [400, 400, 000, 000],
                        ],
                        dtype=torch.int16,
                    ),
                },
            ],
            # fmt: off
            [
                False, False, # ACTIVE SUBENV
                True, False, True, # MASK
                False, False, False, False, False # PAD
            ]
            # fmt: on
        ),
        # Intermediate state, grid active (toggle flag 1)
        (
            "env_grid2d_tetrismini",
            [
                [0, 1, [0, 0], [0, 1]],
                {
                    0: [2, 1],
                    1: torch.tensor(
                        [
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [400, 400, 000, 000],
                            [400, 400, 000, 000],
                        ],
                        dtype=torch.int16,
                    ),
                },
            ],
            # fmt: off
            [
                False, False, # ACTIVE SUBENV
                False, True, True, # MASK
                False, False, False, False, False # PAD
            ]
            # fmt: on
        ),
        # Intermediate state, grid active (toggle flag 0)
        (
            "env_grid2d_tetrismini",
            [
                [0, 0, [0, 0], [0, 1]],
                {
                    0: [2, 1],
                    1: torch.tensor(
                        [
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [400, 400, 000, 000],
                            [400, 400, 000, 000],
                        ],
                        dtype=torch.int16,
                    ),
                },
            ],
            # fmt: off
            [
                True, False, # ACTIVE SUBENV
                False, False, True, # MASK
                False, False, False, False, False # PAD
            ]
            # fmt: on
        ),
    ],
)
def test__get_mask_invalid_actions_backward__returns_expected(
    env, state, mask_exp, request
):
    env = request.getfixturevalue(env)
    mask = env.get_mask_invalid_actions_backward(state, done=False)
    assert mask == mask_exp


@pytest.mark.parametrize(
    "env, state, mask_exp",
    [
        # Intermediate state, no active subenv
        (
            "env_grid2d_tetrismini",
            [
                [-1, 0, [1, 1], [0, 1]],
                {
                    0: [2, 1],
                    1: torch.tensor(
                        [
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [400, 400, 000, 000],
                            [400, 400, 000, 000],
                        ],
                        dtype=torch.int16,
                    ),
                },
            ],
            # fmt: off
            [
                False, False, # ACTIVE SUBENV
                True, True, False, # MASK
                False, False, False, False, False # PAD
            ]
            # fmt: on
        ),
        # Intermediate state, grid active (after subenv action)
        (
            "env_grid2d_tetrismini",
            [
                [0, 0, [1, 1], [0, 1]],
                {
                    0: [2, 1],
                    1: torch.tensor(
                        [
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [400, 400, 000, 000],
                            [400, 400, 000, 000],
                        ],
                        dtype=torch.int16,
                    ),
                },
            ],
            # fmt: off
            [
                False, False, # ACTIVE SUBENV
                False, True, True, # MASK
                False, False, False, False, False # PAD
            ]
            # fmt: on
        ),
    ],
)
def test__get_mask_invalid_actions_forward__all_subenvs_done(
    env, state, mask_exp, request
):
    env = request.getfixturevalue(env)
    # Set sub-environments as done
    for subenv in env.subenvs:
        subenv.done = True
    mask = env.get_mask_invalid_actions_forward(state, done=False)
    assert mask == mask_exp


@pytest.mark.parametrize(
    "env, state, done, mask_exp",
    [
        # All done
        (
            "env_grid2d_tetrismini",
            [
                [-1, 0, [1, 1], [0, 1]],
                {
                    0: [2, 1],
                    1: torch.tensor(
                        [
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [400, 400, 000, 000],
                            [400, 400, 000, 000],
                        ],
                        dtype=torch.int16,
                    ),
                },
            ],
            True,
            # fmt: off
            [
                False, False, # ACTIVE SUBENV
                True, True, False, # MASK
                False, False, False, False, False # PAD
            ]
            # fmt: on
        ),
        # Subenvs done, but not set.
        (
            "env_grid2d_tetrismini",
            [
                [-1, 0, [1, 1], [0, 1]],
                {
                    0: [2, 1],
                    1: torch.tensor(
                        [
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [400, 400, 000, 000],
                            [400, 400, 000, 000],
                        ],
                        dtype=torch.int16,
                    ),
                },
            ],
            False,
            # fmt: off
            [
                False, False, # ACTIVE SUBENV
                False, False, True, # MASK
                False, False, False, False, False # PAD
            ]
            # fmt: on
        ),
        # Grid active and done (toggle flag 0)
        (
            "env_grid2d_tetrismini",
            [
                [0, 0, [1, 1], [0, 1]],
                {
                    0: [2, 1],
                    1: torch.tensor(
                        [
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [400, 400, 000, 000],
                            [400, 400, 000, 000],
                        ],
                        dtype=torch.int16,
                    ),
                },
            ],
            False,
            # fmt: off
            [
                True, False, # ACTIVE SUBENV
                True, True, False, # MASK
                False, False, False, False, False # PAD
            ]
            # fmt: on
        ),
    ],
)
def test__get_mask_invalid_actions_backward__all_subenvs_done(
    env, state, done, mask_exp, request
):
    env = request.getfixturevalue(env)
    env.set_state(state, done)
    mask = env.get_mask_invalid_actions_backward()
    assert mask == mask_exp


@pytest.mark.parametrize(
    "env, mask, active_subenv, mask_core",
    [
        (
            "env_grid2d_tetrismini",
            # fmt: off
            [
                False, False, # ACTIVE SUBENV
                True, True, False, # CORE MASK
                False, False, False, False, False # PAD
            ],
            # fmt: on
            -1,
            [True, True, False],
        ),
        (
            "env_grid2d_tetrismini",
            # fmt: off
            [
                True, False, # ACTIVE SUBENV
                True, True, False, # MASK
                False, False, False, False, False # PAD
            ],
            # fmt: on
            0,
            [True, True, False],
        ),
        (
            "env_grid2d_tetrismini",
            # fmt: off
            [
                False, True, # ACTIVE SUBENV
                False, False, False, False, False, False, False, False # MASK
            ],
            # fmt: on
            1,
            [False, False, False, False, False, False, False, False],
        ),
        (
            "env_grid2d_tetrismini",
            torch.tensor(
                [
                    # fmt: off
                    [
                    False, False, # ACTIVE SUBENV
                    True, True, False, # CORE MASK
                    False, False, False, False, False # PAD
                    ],
                    [
                    False, False, # ACTIVE SUBENV
                    False, True, True, # CORE MASK
                    False, False, False, False, False # PAD
                    ],
                    [
                    False, False, # ACTIVE SUBENV
                    False, False, True, # CORE MASK
                    False, False, False, False, False # PAD
                    ],
                    # fmt: on
                ],
                dtype=torch.bool,
            ),
            -1,
            torch.tensor(
                [
                    [True, True, False],
                    [False, True, True],
                    [False, False, True],
                ],
                dtype=torch.bool,
            ),
        ),
        (
            "env_grid2d_tetrismini",
            torch.tensor(
                [
                    # fmt: off
                    [
                    False, True, # ACTIVE SUBENV
                    False, False, False, False, False, False, False, False # MASK
                    ],
                    [
                    False, True, # ACTIVE SUBENV
                    True, False, False, False, False, False, False, False # MASK
                    ],
                    # fmt: on
                ],
                dtype=torch.bool,
            ),
            1,
            torch.tensor(
                [
                    [False, False, False, False, False, False, False, False],
                    [True, False, False, False, False, False, False, False],
                ],
                dtype=torch.bool,
            ),
        ),
    ],
)
def test__extract_core_mask__returns_expected(
    env, mask, active_subenv, mask_core, request
):
    env = request.getfixturevalue(env)
    if isinstance(mask, list):
        assert mask_core == env._extract_core_mask(mask, active_subenv)
    else:
        assert torch.equal(mask_core, env._extract_core_mask(mask, active_subenv))


@pytest.mark.repeat(10)
@pytest.mark.parametrize(
    "env",
    [
        "env_grid2d_tetrismini",
        "env_cube_tetris",
        "env_cube_tetris_grid",
        "env_two_grids",
        "env_three_cubes",
        "env_cube2d_cube3d",
        "env_two_cubes2d_one_cube3d",
    ],
)
def test__step_random__does_not_crash_from_source(env, request):
    """
    Very low bar test...
    """
    env = request.getfixturevalue(env)
    env.reset()
    state_next, action, valid = env.step_random()
    assert True


@pytest.mark.parametrize(
    "env, state, parents_exp, parent_actions_exp",
    [
        # Source
        (
            "env_grid2d_tetrismini",
            [
                [-1, 0, [0, 0], [0, 1]],
                {
                    0: [0, 0],
                    1: torch.tensor(
                        [
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                        ],
                        dtype=torch.int16,
                    ),
                },
            ],
            [],
            [],
        ),
        # Source -> activate grid
        (
            "env_grid2d_tetrismini",
            [
                [0, 1, [0, 0], [0, 1]],
                {
                    0: [0, 0],
                    1: torch.tensor(
                        [
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                        ],
                        dtype=torch.int16,
                    ),
                },
            ],
            [
                [
                    -1,
                    0,
                    [0, 0],
                    {
                        0: [0, 0],
                        1: torch.tensor(
                            [
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
            ],
            [(-1, 0, 0, 0)],
        ),
        # Source -> activate tetris
        (
            "env_grid2d_tetrismini",
            [
                [1, 1, [0, 0], [0, 1]],
                {
                    0: [0, 0],
                    1: torch.tensor(
                        [
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                        ],
                        dtype=torch.int16,
                    ),
                },
            ],
            [
                [
                    [-1, 0, [0, 0], [0, 1]],
                    {
                        0: [0, 0],
                        1: torch.tensor(
                            [
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
            ],
            [(-1, 1, 0, 0)],
        ),
        # Intermediate state, no active subenv, both states no source
        (
            "env_grid2d_tetrismini",
            [
                [-1, 0, [0, 0], [0, 1]],
                {
                    0: [2, 1],
                    1: torch.tensor(
                        [
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [400, 400, 000, 000],
                            [400, 400, 000, 000],
                        ],
                        dtype=torch.int16,
                    ),
                },
            ],
            [
                [
                    [0, 0, [0, 0], [0, 1]],
                    {
                        0: [2, 1],
                        1: torch.tensor(
                            [
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [400, 400, 000, 000],
                                [400, 400, 000, 000],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
                [
                    [1, 0, [0, 0], [0, 1]],
                    {
                        0: [2, 1],
                        1: torch.tensor(
                            [
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [400, 400, 000, 000],
                                [400, 400, 000, 000],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
            ],
            [(-1, 0, 0, 0), (-1, 1, 0, 0)],
        ),
        # Intermediate state, no active subenv, tetris is source
        (
            "env_grid2d_tetrismini",
            [
                [-1, 0, [0, 0], [0, 1]],
                {
                    0: [2, 1],
                    1: torch.tensor(
                        [
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                        ],
                        dtype=torch.int16,
                    ),
                },
            ],
            [
                [
                    [-1, 0, [0, 0], [0, 1]],
                    {
                        0: [2, 1],
                        1: torch.tensor(
                            [
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
            ],
            [(-1, 0, 0, 0)],
        ),
        # Intermediate state, grid active (toggle flag 1)
        (
            "env_grid2d_tetrismini",
            [
                [0, 1, [0, 0], [0, 1]],
                {
                    0: [2, 1],
                    1: torch.tensor(
                        [
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [400, 400, 000, 000],
                            [400, 400, 000, 000],
                        ],
                        dtype=torch.int16,
                    ),
                },
            ],
            [
                [
                    [-1, 0, [0, 0], [0, 1]],
                    {
                        0: [2, 1],
                        1: torch.tensor(
                            [
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [400, 400, 000, 000],
                                [400, 400, 000, 000],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
            ],
            [(-1, 0, 0, 0)],
        ),
        # Intermediate state, grid active (toggle flag 0)
        (
            "env_grid2d_tetrismini",
            [
                [0, 0, [0, 0], [0, 1]],
                {
                    0: [2, 1],
                    1: torch.tensor(
                        [
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [400, 400, 000, 000],
                            [400, 400, 000, 000],
                        ],
                        dtype=torch.int16,
                    ),
                },
            ],
            [
                [
                    [0, 1, [0, 0], [0, 1]],
                    {
                        0: [1, 1],
                        1: torch.tensor(
                            [
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [400, 400, 000, 000],
                                [400, 400, 000, 000],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
                [
                    [0, 1, [0, 0], [0, 1]],
                    {
                        0: [2, 0],
                        1: torch.tensor(
                            [
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [400, 400, 000, 000],
                                [400, 400, 000, 000],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
            ],
            [(0, 1, 0, 0), (0, 0, 1, 0)],
        ),
    ],
)
def test__get_parents__returns_expected(
    env, state, parents_exp, parent_actions_exp, request
):
    env = request.getfixturevalue(env)

    parents, parent_actions = env.get_parents(state, done=False)

    # Create dictionaries of parent_action: parent for comparison
    parents_actions_exp_dict = {}
    for parent, action in zip(parents_exp, parent_actions_exp):
        parents_actions_exp_dict[action] = tuple(parent.copy())
    parents_actions_dict = {}
    for parent, action in zip(parents, parent_actions):
        parents_actions_dict[action] = tuple(parent.copy())

    assert sorted(parents_actions_exp_dict) == sorted(parents_actions_dict)


@pytest.mark.parametrize(
    "env, state, valid_actions",
    [
        (
            "env_grid2d_tetrismini",
            # Source
            [
                [-1, 0, [0, 0], [0, 1]],
                {
                    0: [0, 0],
                    1: torch.tensor(
                        [
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                        ],
                        dtype=torch.int16,
                    ),
                },
            ],
            [(-1, 0, 0, 0), (-1, 1, 0, 0)],
        ),
    ],
)
def test__get_valid_actions__forward_returns_expected(
    env, state, valid_actions, request
):
    env = request.getfixturevalue(env)
    assert set(valid_actions) == set(
        env.get_valid_actions(state=state, done=False, backward=False)
    )


@pytest.mark.parametrize(
    "env, state, valid_actions",
    [
        (
            "env_grid2d_tetrismini",
            # Active grid and toggle flag 1
            [
                [0, 1, [0, 0], [0, 1]],
                {
                    0: [0, 0],
                    1: torch.tensor(
                        [
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                        ],
                        dtype=torch.int16,
                    ),
                },
            ],
            [(-1, 0, 0, 0)],
        ),
        (
            "env_grid2d_tetrismini",
            # Active tetris and toggle flag 1
            [
                [1, 1, [0, 0], [0, 1]],
                {
                    0: [0, 0],
                    1: torch.tensor(
                        [
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                        ],
                        dtype=torch.int16,
                    ),
                },
            ],
            [(-1, 1, 0, 0)],
        ),
        (
            "env_grid2d_tetrismini",
            # Intermediate state, inactive
            [
                [-1, 0, [0, 0], [0, 1]],
                {
                    0: [1, 2],
                    1: torch.tensor(
                        [
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [100, 100, 100, 100],
                            [000, 400, 400, 000],
                            [000, 400, 400, 000],
                        ],
                        dtype=torch.int16,
                    ),
                },
            ],
            [(-1, 0, 0, 0), (-1, 1, 0, 0)],
        ),
        (
            "env_grid2d_tetrismini",
            # Active grid, toggle flag 0
            [
                [0, 0, [0, 0], [0, 1]],
                {
                    0: [1, 2],
                    1: torch.tensor(
                        [
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [100, 100, 100, 100],
                            [000, 400, 400, 000],
                            [000, 400, 400, 000],
                        ],
                        dtype=torch.int16,
                    ),
                },
            ],
            [(0, 1, 0, 0), (0, 0, 1, 0)],
        ),
        (
            "env_stacks_equal",
            # Active Stack 1, toggle flag 0, Stack 1 is done
            # Only valid action is EOS of grid (there is only one group)
            [
                # fmt: off
                [1, 0, [1, 1], [0, 0]],
                {
                    0: [1, [0.5407, 0.4987], [1, 0]],
                    1: [1, [0.6297, 0.4120], [0, 0]],
                },
                # fmt: on
            ],
            [(0, 1, 0, 0, 0)],
        ),
    ],
)
def test__get_valid_actions__backwards_returns_expected(
    env, state, valid_actions, request
):
    env = request.getfixturevalue(env)
    assert set(valid_actions) == set(
        env.get_valid_actions(state=state, done=False, backward=True)
    )


@pytest.mark.parametrize(
    "env, state",
    [
        (
            "env_grid2d_tetrismini",
            # Active grid and toggle flag 1
            [
                [0, 1, [0, 0], [0, 1]],
                {
                    0: [0, 0],
                    1: torch.tensor(
                        [
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                        ],
                        dtype=torch.int16,
                    ),
                },
            ],
        ),
        (
            "env_grid2d_tetrismini",
            # Active tetris and toggle flag 1
            [
                [1, 1, [0, 0], [0, 1]],
                {
                    0: [0, 0],
                    1: torch.tensor(
                        [
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                        ],
                        dtype=torch.int16,
                    ),
                },
            ],
        ),
        (
            "env_grid2d_tetrismini",
            # Intermediate state, inactive
            [
                [-1, 0, [0, 0], [0, 1]],
                {
                    0: [1, 2],
                    1: torch.tensor(
                        [
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [100, 100, 100, 100],
                            [000, 400, 400, 000],
                            [000, 400, 400, 000],
                        ],
                        dtype=torch.int16,
                    ),
                },
            ],
        ),
        (
            "env_grid2d_tetrismini",
            # Active grid, toggle flag 0
            [
                [0, 0, [0, 0], [0, 1]],
                {
                    0: [1, 2],
                    1: torch.tensor(
                        [
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [100, 100, 100, 100],
                            [000, 400, 400, 000],
                            [000, 400, 400, 000],
                        ],
                        dtype=torch.int16,
                    ),
                },
            ],
        ),
        (
            "env_stacks_equal",
            # Active Stack 1, toggle flag 0, Stack 1 is done
            # Only valid action is EOS of grid (there is only one group)
            [
                # fmt: off
                [1, 0, [1, 1], [0, 0]],
                {
                    0: [1, [0.5407, 0.4987], [1, 0]],
                    1: [1, [0.6297, 0.4120], [0, 0]],
                },
                # fmt: on
            ],
        ),
    ],
)
def test__get_valid_actions__is_consistent_regardless_of_inputs(env, state, request):
    env = request.getfixturevalue(env)
    # Reset environment (state will be source)
    env.reset()
    # Obtain masks by passing state
    mask_f = env.get_mask_invalid_actions_forward(state, done=False)
    mask_b = env.get_mask_invalid_actions_backward(state, done=False)
    # Obtain valid actions by passing mask and state
    valid_actions_f_mask_state = env.get_valid_actions(
        mask=mask_f, state=state, done=False, backward=False
    )
    valid_actions_b_mask_state = env.get_valid_actions(
        mask=mask_b, state=state, done=False, backward=True
    )
    # Obtain valid actions by passing state but not mask. In this case, the
    # mask should be computed from the state inside get_valid_actions.
    valid_actions_f_state = env.get_valid_actions(
        state=state, done=False, backward=False
    )
    valid_actions_b_state = env.get_valid_actions(
        state=state, done=False, backward=True
    )
    # Check that the valid actions are the same in both cases
    assert valid_actions_f_mask_state == valid_actions_f_state
    assert valid_actions_b_mask_state == valid_actions_b_state
    # Set state
    env.set_state(state, done=False)
    # Obtain valid actions without passing neither the mask nor the state.
    # In this case, both the state and the mask should be computed from the
    # environment.
    valid_actions_f = env.get_valid_actions(done=False, backward=False)
    valid_actions_b = env.get_valid_actions(done=False, backward=True)
    # Check that the valid actions are still the same
    assert valid_actions_f_mask_state == valid_actions_f
    assert valid_actions_b_mask_state == valid_actions_b


@pytest.mark.repeat(10)
@pytest.mark.parametrize(
    "env, states",
    [
        (
            "env_grid2d_tetrismini",
            # Two source states
            [
                [
                    [-1, 0, [0, 0], [0, 1]],
                    {
                        0: [0, 0],
                        1: torch.tensor(
                            [
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
                [
                    [-1, 0, [0, 0], [0, 1]],
                    {
                        0: [0, 0],
                        1: torch.tensor(
                            [
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
            ],
        ),
        (
            "env_grid2d_tetrismini",
            # Mixed states
            [
                [
                    [-1, 0, [0, 0], [0, 1]],
                    {
                        0: [0, 0],
                        1: torch.tensor(
                            [
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
                [
                    [0, 1, [0, 0], [0, 1]],
                    {
                        0: [0, 0],
                        1: torch.tensor(
                            [
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
                [
                    [0, 0, [0, 0], [0, 1]],
                    {
                        0: [1, 0],
                        1: torch.tensor(
                            [
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
                [
                    [-1, 0, [0, 0], [0, 1]],
                    {
                        0: [1, 0],
                        1: torch.tensor(
                            [
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
                [
                    [1, 1, [0, 0], [0, 1]],
                    {
                        0: [1, 0],
                        1: torch.tensor(
                            [
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
            ],
        ),
    ],
)
def test__sample_actions_forward__returns_valid_actions(env, states, request):
    env = request.getfixturevalue(env)
    n_states = len(states)
    # Get masks
    masks = tbool(
        [env.get_mask_invalid_actions_forward(s) for s in states], device=env.device
    )
    # Build policy outputs
    params = env.random_distr_params
    policy_outputs = torch.tile(env.get_policy_output(params), dims=(n_states, 1))
    # Sample actions
    actions, _ = env.sample_actions_batch(
        policy_outputs, masks, states, is_backward=False
    )
    # Sample actions are valid
    for state, action in zip(states, actions):
        assert action in env.get_valid_actions(state=state, done=False, backward=False)


@pytest.mark.repeat(10)
@pytest.mark.parametrize(
    "env, states",
    [
        (
            "env_grid2d_tetrismini",
            # Two done states
            [
                [
                    [-1, 0, [1, 1], [0, 1]],
                    {
                        0: [1, 2],
                        1: torch.tensor(
                            [
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [400, 400, 000, 000],
                                [400, 400, 000, 000],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
                [
                    [-1, 0, [1, 1], [0, 1]],
                    {
                        0: [0, 0],
                        1: torch.tensor(
                            [
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
            ],
        ),
        (
            "env_grid2d_tetrismini",
            # Mixed states
            [
                [
                    [-1, 0, [1, 0], [0, 1]],
                    {
                        0: [0, 0],
                        1: torch.tensor(
                            [
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
                [
                    [0, 1, [0, 0], [0, 1]],
                    {
                        0: [0, 0],
                        1: torch.tensor(
                            [
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
                [
                    [0, 0, [0, 0], [0, 1]],
                    {
                        0: [1, 0],
                        1: torch.tensor(
                            [
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
                [
                    [-1, 0, [0, 0], [0, 1]],
                    {
                        0: [1, 0],
                        1: torch.tensor(
                            [
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
                [
                    [1, 1, [0, 0], [0, 1]],
                    {
                        0: [1, 0],
                        1: torch.tensor(
                            [
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
            ],
        ),
    ],
)
def test__sample_actions_backward__returns_valid_actions(env, states, request):
    env = request.getfixturevalue(env)
    n_states = len(states)
    # Get masks
    masks = tbool(
        [env.get_mask_invalid_actions_backward(s) for s in states], device=env.device
    )
    # Build policy outputs
    params = env.random_distr_params
    policy_outputs = torch.tile(env.get_policy_output(params), dims=(n_states, 1))
    # Sample actions
    actions, _ = env.sample_actions_batch(
        policy_outputs, masks, states, is_backward=True
    )
    # Sample actions are valid
    for state, action in zip(states, actions):
        assert action in env.get_valid_actions(state=state, done=False, backward=True)


@pytest.mark.repeat(10)
@pytest.mark.parametrize(
    "env, states",
    [
        (
            "env_grid2d_tetrismini",
            # Two source states
            [
                [
                    [-1, 0, [0, 0], [0, 1]],
                    {
                        0: [0, 0],
                        1: torch.tensor(
                            [
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
                [
                    [-1, 0, [0, 0], [0, 1]],
                    {
                        0: [0, 0],
                        1: torch.tensor(
                            [
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
            ],
        ),
        (
            "env_grid2d_tetrismini",
            # Mixed states
            [
                [
                    [-1, 0, [0, 0], [0, 1]],
                    {
                        0: [0, 0],
                        1: torch.tensor(
                            [
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
                [
                    [0, 1, [0, 0], [0, 1]],
                    {
                        0: [0, 0],
                        1: torch.tensor(
                            [
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
                [
                    [0, 0, [0, 0], [0, 1]],
                    {
                        0: [1, 0],
                        1: torch.tensor(
                            [
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
                [
                    [-1, 0, [0, 0], [0, 1]],
                    {
                        0: [1, 0],
                        1: torch.tensor(
                            [
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
                [
                    [1, 1, [0, 0], [0, 1]],
                    {
                        0: [1, 0],
                        1: torch.tensor(
                            [
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
            ],
        ),
    ],
)
def test__get_logprobs_forward__all_finite(env, states, request):
    env = request.getfixturevalue(env)
    n_states = len(states)
    # Get masks
    masks = tbool(
        [env.get_mask_invalid_actions_forward(s) for s in states], device=env.device
    )
    # Build policy outputs
    params = env.random_distr_params
    policy_outputs = torch.tile(env.get_policy_output(params), dims=(n_states, 1))
    # Sample actions
    actions, _ = env.sample_actions_batch(
        policy_outputs, masks, states, is_backward=False
    )

    actions_torch = torch.tensor(actions)
    logprobs = env.get_logprobs(
        policy_outputs=policy_outputs,
        actions=actions_torch,
        mask=masks,
        states_from=states,
        is_backward=False,
    )
    assert torch.all(torch.isfinite(logprobs))


@pytest.mark.repeat(10)
@pytest.mark.parametrize(
    "env, states",
    [
        (
            "env_grid2d_tetrismini",
            # Two done states
            [
                [
                    [-1, 0, [1, 1], [0, 1]],
                    {
                        0: [1, 2],
                        1: torch.tensor(
                            [
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [400, 400, 000, 000],
                                [400, 400, 000, 000],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
                [
                    [-1, 0, [1, 1], [0, 1]],
                    {
                        0: [0, 0],
                        1: torch.tensor(
                            [
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
            ],
        ),
        (
            "env_grid2d_tetrismini",
            # Mixed states
            [
                [
                    [-1, 0, [1, 0], [0, 1]],
                    {
                        0: [0, 0],
                        1: torch.tensor(
                            [
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
                [
                    [0, 1, [0, 0], [0, 1]],
                    {
                        0: [0, 0],
                        1: torch.tensor(
                            [
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
                [
                    [0, 0, [0, 0], [0, 1]],
                    {
                        0: [1, 0],
                        1: torch.tensor(
                            [
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
                [
                    [-1, 0, [0, 0], [0, 1]],
                    {
                        0: [1, 0],
                        1: torch.tensor(
                            [
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
                [
                    [1, 1, [0, 0], [0, 1]],
                    {
                        0: [1, 0],
                        1: torch.tensor(
                            [
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
            ],
        ),
    ],
)
def test__get_logprobs_backward__all_finite(env, states, request):
    env = request.getfixturevalue(env)
    n_states = len(states)
    # Get masks
    masks = tbool(
        [env.get_mask_invalid_actions_backward(s) for s in states], device=env.device
    )
    # Build policy outputs
    params = env.random_distr_params
    policy_outputs = torch.tile(env.get_policy_output(params), dims=(n_states, 1))
    # Sample actions
    actions, _ = env.sample_actions_batch(
        policy_outputs, masks, states, is_backward=True
    )

    actions_torch = torch.tensor(actions)
    logprobs = env.get_logprobs(
        policy_outputs=policy_outputs,
        actions=actions_torch,
        mask=masks,
        states_from=states,
        is_backward=True,
    )
    assert torch.all(torch.isfinite(logprobs))


@pytest.mark.parametrize(
    "env, state, action",
    [
        (
            "env_grid2d_tetrismini",
            [
                [1, 0, [0, 0], [0, 1]],
                {
                    0: [0, 0],
                    1: torch.tensor(
                        [
                            [000, 000, 000, 000],
                            [000, 000, 100, 000],
                            [000, 000, 100, 000],
                            [400, 400, 100, 000],
                            [400, 400, 100, 000],
                        ],
                        dtype=torch.int16,
                    ),
                },
            ],
            (1, 1, 0, 2),
        ),
    ],
)
def test__get_logprobs_backward__is_finite(env, state, action, request):
    env = request.getfixturevalue(env)
    masks = torch.unsqueeze(
        tbool(
            env.get_mask_invalid_actions_backward(state, done=False), device=env.device
        ),
        0,
    )
    policy_outputs = torch.unsqueeze(env.random_policy_output, 0)
    actions_torch = torch.unsqueeze(torch.tensor(action), 0)
    logprobs = env.get_logprobs(
        policy_outputs=policy_outputs,
        actions=actions_torch,
        mask=masks,
        states_from=[state],
        is_backward=True,
    )
    assert torch.all(torch.isfinite(logprobs))


@pytest.mark.parametrize(
    "env, states, actions",
    [
        (
            "env_grid2d_tetrismini",
            [
                [
                    [1, 1, [0, 0], [0, 1]],
                    {
                        0: [0, 0],
                        1: torch.tensor(
                            [
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
                [
                    [1, 0, [0, 0], [0, 1]],
                    {
                        0: [0, 0],
                        1: torch.tensor(
                            [
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [400, 400, 0, 0],
                                [400, 400, 0, 0],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
                [
                    [-1, 0, [0, 0], [0, 1]],
                    {
                        0: [0, 0],
                        1: torch.tensor(
                            [
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [400, 400, 0, 0],
                                [400, 400, 0, 0],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
                [
                    [1, 1, [0, 0], [0, 1]],
                    {
                        0: [0, 0],
                        1: torch.tensor(
                            [
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [400, 400, 0, 0],
                                [400, 400, 0, 0],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
                [
                    [1, 0, [0, 0], [0, 1]],
                    {
                        0: [0, 0],
                        1: torch.tensor(
                            [
                                [0, 0, 0, 0],
                                [0, 0, 100, 0],
                                [0, 0, 100, 0],
                                [400, 400, 100, 0],
                                [400, 400, 100, 0],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
                [
                    [-1, 0, [0, 0], [0, 1]],
                    {
                        0: [0, 0],
                        1: torch.tensor(
                            [
                                [0, 0, 0, 0],
                                [0, 0, 100, 0],
                                [0, 0, 100, 0],
                                [400, 400, 100, 0],
                                [400, 400, 100, 0],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
                [
                    [0, 1, [0, 0], [0, 1]],
                    {
                        0: [0, 0],
                        1: torch.tensor(
                            [
                                [0, 0, 0, 0],
                                [0, 0, 100, 0],
                                [0, 0, 100, 0],
                                [400, 400, 100, 0],
                                [400, 400, 100, 0],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
                [
                    [0, 0, [0, 0], [0, 1]],
                    {
                        0: [0, 1],
                        1: torch.tensor(
                            [
                                [0, 0, 0, 0],
                                [0, 0, 100, 0],
                                [0, 0, 100, 0],
                                [400, 400, 100, 0],
                                [400, 400, 100, 0],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
                [
                    [-1, 0, [0, 0], [0, 1]],
                    {
                        0: [0, 1],
                        1: torch.tensor(
                            [
                                [0, 0, 0, 0],
                                [0, 0, 100, 0],
                                [0, 0, 100, 0],
                                [400, 400, 100, 0],
                                [400, 400, 100, 0],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
                [
                    [1, 1, [0, 0], [0, 1]],
                    {
                        0: [0, 1],
                        1: torch.tensor(
                            [
                                [0, 0, 0, 0],
                                [0, 0, 100, 0],
                                [0, 0, 100, 0],
                                [400, 400, 100, 0],
                                [400, 400, 100, 0],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
                [
                    [1, 0, [0, 0], [0, 1]],
                    {
                        0: [0, 1],
                        1: torch.tensor(
                            [
                                [0, 0, 0, 0],
                                [0, 0, 100, 101],
                                [0, 0, 100, 101],
                                [400, 400, 100, 101],
                                [400, 400, 100, 101],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
                [
                    [-1, 0, [0, 0], [0, 1]],
                    {
                        0: [0, 1],
                        1: torch.tensor(
                            [
                                [0, 0, 0, 0],
                                [0, 0, 100, 101],
                                [0, 0, 100, 101],
                                [400, 400, 100, 101],
                                [400, 400, 100, 101],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
                [
                    [1, 1, [0, 0], [0, 1]],
                    {
                        0: [0, 1],
                        1: torch.tensor(
                            [
                                [0, 0, 0, 0],
                                [0, 0, 100, 101],
                                [0, 0, 100, 101],
                                [400, 400, 100, 101],
                                [400, 400, 100, 101],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
                [
                    [1, 0, [0, 0], [0, 1]],
                    {
                        0: [0, 1],
                        1: torch.tensor(
                            [
                                [0, 0, 0, 0],
                                [401, 401, 100, 101],
                                [401, 401, 100, 101],
                                [400, 400, 100, 101],
                                [400, 400, 100, 101],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
                [
                    [-1, 0, [0, 0], [0, 1]],
                    {
                        0: [0, 1],
                        1: torch.tensor(
                            [
                                [0, 0, 0, 0],
                                [401, 401, 100, 101],
                                [401, 401, 100, 101],
                                [400, 400, 100, 101],
                                [400, 400, 100, 101],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
                [
                    [1, 1, [0, 0], [0, 1]],
                    {
                        0: [0, 1],
                        1: torch.tensor(
                            [
                                [0, 0, 0, 0],
                                [401, 401, 100, 101],
                                [401, 401, 100, 101],
                                [400, 400, 100, 101],
                                [400, 400, 100, 101],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
                [
                    [1, 0, [0, 1], [0, 1]],
                    {
                        0: [0, 1],
                        1: torch.tensor(
                            [
                                [0, 0, 0, 0],
                                [401, 401, 100, 101],
                                [401, 401, 100, 101],
                                [400, 400, 100, 101],
                                [400, 400, 100, 101],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
                [
                    [-1, 0, [0, 1], [0, 1]],
                    {
                        0: [0, 1],
                        1: torch.tensor(
                            [
                                [0, 0, 0, 0],
                                [401, 401, 100, 101],
                                [401, 401, 100, 101],
                                [400, 400, 100, 101],
                                [400, 400, 100, 101],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
                [
                    [0, 1, [0, 1], [0, 1]],
                    {
                        0: [0, 1],
                        1: torch.tensor(
                            [
                                [0, 0, 0, 0],
                                [401, 401, 100, 101],
                                [401, 401, 100, 101],
                                [400, 400, 100, 101],
                                [400, 400, 100, 101],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
                [
                    [0, 0, [1, 1], [0, 1]],
                    {
                        0: [0, 1],
                        1: torch.tensor(
                            [
                                [0, 0, 0, 0],
                                [401, 401, 100, 101],
                                [401, 401, 100, 101],
                                [400, 400, 100, 101],
                                [400, 400, 100, 101],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
                [
                    [-1, 0, [1, 1], [0, 1]],
                    {
                        0: [0, 1],
                        1: torch.tensor(
                            [
                                [0, 0, 0, 0],
                                [401, 401, 100, 101],
                                [401, 401, 100, 101],
                                [400, 400, 100, 101],
                                [400, 400, 100, 101],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
                [
                    [-1, 0, [1, 1], [0, 1]],
                    {
                        0: [0, 1],
                        1: torch.tensor(
                            [
                                [0, 0, 0, 0],
                                [401, 401, 100, 101],
                                [401, 401, 100, 101],
                                [400, 400, 100, 101],
                                [400, 400, 100, 101],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
            ],
            torch.tensor(
                [
                    [-1.0, 1.0, 0.0, 0.0],
                    [1.0, 4.0, 0.0, 0.0],
                    [-1.0, 1.0, 0.0, 0.0],
                    [-1.0, 1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0, 2.0],
                    [-1.0, 1.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [-1.0, 0.0, 0.0, 0.0],
                    [-1.0, 1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0, 3.0],
                    [-1.0, 1.0, 0.0, 0.0],
                    [-1.0, 1.0, 0.0, 0.0],
                    [1.0, 4.0, 0.0, 0.0],
                    [-1.0, 1.0, 0.0, 0.0],
                    [-1.0, 1.0, 0.0, 0.0],
                    [1.0, -1.0, -1.0, -1.0],
                    [-1.0, 1.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0, 0.0],
                    [-1.0, -1.0, -1.0, -1.0],
                ]
            ),
        ),
        (
            "env_grid2d_tetrismini",
            [
                [
                    [0, 1, [0, 0], [0, 1]],
                    {
                        0: [0, 0],
                        1: torch.tensor(
                            [
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
                [
                    [0, 0, [0, 0], [0, 1]],
                    {
                        0: [0, 1],
                        1: torch.tensor(
                            [
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
                [
                    [-1, 0, [0, 0], [0, 1]],
                    {
                        0: [0, 1],
                        1: torch.tensor(
                            [
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
                [
                    [1, 1, [0, 0], [0, 1]],
                    {
                        0: [0, 1],
                        1: torch.tensor(
                            [
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
                [
                    [1, 0, [0, 0], [0, 1]],
                    {
                        0: [0, 1],
                        1: torch.tensor(
                            [
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 400, 400],
                                [0, 0, 400, 400],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
                [
                    [-1, 0, [0, 0], [0, 1]],
                    {
                        0: [0, 1],
                        1: torch.tensor(
                            [
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 400, 400],
                                [0, 0, 400, 400],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
                [
                    [1, 1, [0, 0], [0, 1]],
                    {
                        0: [0, 1],
                        1: torch.tensor(
                            [
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 400, 400],
                                [0, 0, 400, 400],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
                [
                    [1, 0, [0, 0], [0, 1]],
                    {
                        0: [0, 1],
                        1: torch.tensor(
                            [
                                [0, 0, 0, 0],
                                [100, 0, 0, 0],
                                [100, 0, 0, 0],
                                [100, 0, 400, 400],
                                [100, 0, 400, 400],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
                [
                    [-1, 0, [0, 0], [0, 1]],
                    {
                        0: [0, 1],
                        1: torch.tensor(
                            [
                                [0, 0, 0, 0],
                                [100, 0, 0, 0],
                                [100, 0, 0, 0],
                                [100, 0, 400, 400],
                                [100, 0, 400, 400],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
                [
                    [1, 1, [0, 0], [0, 1]],
                    {
                        0: [0, 1],
                        1: torch.tensor(
                            [
                                [0, 0, 0, 0],
                                [100, 0, 0, 0],
                                [100, 0, 0, 0],
                                [100, 0, 400, 400],
                                [100, 0, 400, 400],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
                [
                    [1, 0, [0, 1], [0, 1]],
                    {
                        0: [0, 1],
                        1: torch.tensor(
                            [
                                [0, 0, 0, 0],
                                [100, 0, 0, 0],
                                [100, 0, 0, 0],
                                [100, 0, 400, 400],
                                [100, 0, 400, 400],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
                [
                    [-1, 0, [0, 1], [0, 1]],
                    {
                        0: [0, 1],
                        1: torch.tensor(
                            [
                                [0, 0, 0, 0],
                                [100, 0, 0, 0],
                                [100, 0, 0, 0],
                                [100, 0, 400, 400],
                                [100, 0, 400, 400],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
                [
                    [0, 1, [0, 1], [0, 1]],
                    {
                        0: [0, 1],
                        1: torch.tensor(
                            [
                                [0, 0, 0, 0],
                                [100, 0, 0, 0],
                                [100, 0, 0, 0],
                                [100, 0, 400, 400],
                                [100, 0, 400, 400],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
                [
                    [0, 0, [0, 1], [0, 1]],
                    {
                        0: [0, 2],
                        1: torch.tensor(
                            [
                                [0, 0, 0, 0],
                                [100, 0, 0, 0],
                                [100, 0, 0, 0],
                                [100, 0, 400, 400],
                                [100, 0, 400, 400],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
                [
                    [-1, 0, [0, 1], [0, 1]],
                    {
                        0: [0, 2],
                        1: torch.tensor(
                            [
                                [0, 0, 0, 0],
                                [100, 0, 0, 0],
                                [100, 0, 0, 0],
                                [100, 0, 400, 400],
                                [100, 0, 400, 400],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
                [
                    [0, 1, [0, 1], [0, 1]],
                    {
                        0: [0, 2],
                        1: torch.tensor(
                            [
                                [0, 0, 0, 0],
                                [100, 0, 0, 0],
                                [100, 0, 0, 0],
                                [100, 0, 400, 400],
                                [100, 0, 400, 400],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
                [
                    [0, 0, [0, 1], [0, 1]],
                    {
                        0: [1, 2],
                        1: torch.tensor(
                            [
                                [0, 0, 0, 0],
                                [100, 0, 0, 0],
                                [100, 0, 0, 0],
                                [100, 0, 400, 400],
                                [100, 0, 400, 400],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
                [
                    [-1, 0, [0, 1], [0, 1]],
                    {
                        0: [1, 2],
                        1: torch.tensor(
                            [
                                [0, 0, 0, 0],
                                [100, 0, 0, 0],
                                [100, 0, 0, 0],
                                [100, 0, 400, 400],
                                [100, 0, 400, 400],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
                [
                    [0, 1, [0, 1], [0, 1]],
                    {
                        0: [1, 2],
                        1: torch.tensor(
                            [
                                [0, 0, 0, 0],
                                [100, 0, 0, 0],
                                [100, 0, 0, 0],
                                [100, 0, 400, 400],
                                [100, 0, 400, 400],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
                [
                    [0, 0, [0, 1], [0, 1]],
                    {
                        0: [2, 2],
                        1: torch.tensor(
                            [
                                [0, 0, 0, 0],
                                [100, 0, 0, 0],
                                [100, 0, 0, 0],
                                [100, 0, 400, 400],
                                [100, 0, 400, 400],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
                [
                    [-1, 0, [0, 1], [0, 1]],
                    {
                        0: [2, 2],
                        1: torch.tensor(
                            [
                                [0, 0, 0, 0],
                                [100, 0, 0, 0],
                                [100, 0, 0, 0],
                                [100, 0, 400, 400],
                                [100, 0, 400, 400],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
                [
                    [0, 1, [0, 1], [0, 1]],
                    {
                        0: [2, 2],
                        1: torch.tensor(
                            [
                                [0, 0, 0, 0],
                                [100, 0, 0, 0],
                                [100, 0, 0, 0],
                                [100, 0, 400, 400],
                                [100, 0, 400, 400],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
                [
                    [0, 0, [1, 1], [0, 1]],
                    {
                        0: [2, 2],
                        1: torch.tensor(
                            [
                                [0, 0, 0, 0],
                                [100, 0, 0, 0],
                                [100, 0, 0, 0],
                                [100, 0, 400, 400],
                                [100, 0, 400, 400],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
                [
                    [-1, 0, [1, 1], [0, 1]],
                    {
                        0: [2, 2],
                        1: torch.tensor(
                            [
                                [0, 0, 0, 0],
                                [100, 0, 0, 0],
                                [100, 0, 0, 0],
                                [100, 0, 400, 400],
                                [100, 0, 400, 400],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
                [
                    [-1, 0, [1, 1], [0, 1]],
                    {
                        0: [2, 2],
                        1: torch.tensor(
                            [
                                [0, 0, 0, 0],
                                [100, 0, 0, 0],
                                [100, 0, 0, 0],
                                [100, 0, 400, 400],
                                [100, 0, 400, 400],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
            ],
            torch.tensor(
                [
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [-1.0, 0.0, 0.0, 0.0],
                    [-1.0, 1.0, 0.0, 0.0],
                    [1.0, 4.0, 0.0, 2.0],
                    [-1.0, 1.0, 0.0, 0.0],
                    [-1.0, 1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0, 0.0],
                    [-1.0, 1.0, 0.0, 0.0],
                    [-1.0, 1.0, 0.0, 0.0],
                    [1.0, -1.0, -1.0, -1.0],
                    [-1.0, 1.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [-1.0, 0.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0, 0.0],
                    [-1.0, -1.0, -1.0, -1.0],
                ]
            ),
        ),
    ],
)
def test__get_logprobs_backward__one_batched_trajectory_all_finite(
    env, states, actions, request
):
    env = request.getfixturevalue(env)
    n_states = len(states)
    # Get masks
    masks = [env.get_mask_invalid_actions_backward(s) for s in states[:-1]]
    masks += [env.get_mask_invalid_actions_backward(states[-1], done=True)]
    masks = tbool(masks, device=env.device)
    # Build policy outputs
    params = env.random_distr_params
    policy_outputs = torch.tile(env.get_policy_output(params), dims=(n_states, 1))

    logprobs = env.get_logprobs(
        policy_outputs=policy_outputs,
        actions=actions,
        mask=masks,
        states_from=states,
        is_backward=True,
    )
    assert torch.all(torch.isfinite(logprobs))


@pytest.mark.repeat(10)
@pytest.mark.parametrize(
    "env",
    [
        "env_grid2d_tetrismini",
        "env_cube_tetris",
        "env_cube_tetris_grid",
        "env_two_grids",
        "env_three_cubes",
        "env_cube2d_cube3d",
        "env_two_cubes2d_one_cube3d",
    ],
)
def test__trajectory_random__does_not_crash_and_reaches_done(env, request):
    """
    Raising the bar...
    """
    env = request.getfixturevalue(env)
    env.reset()
    env.trajectory_random()
    assert env.done


@pytest.mark.repeat(10)
@pytest.mark.parametrize(
    "env",
    [
        "env_grid2d_tetrismini",
        "env_cube_tetris",
        "env_cube_tetris_grid",
        "env_two_grids",
        "env_three_cubes",
        "env_cube2d_cube3d",
        "env_two_cubes2d_one_cube3d",
    ],
)
def test__trajectory_bacwards_random__does_not_crash_and_reaches_source(env, request):
    """
    And backwards!
    """
    env = request.getfixturevalue(env)
    env.reset()
    env.trajectory_random()
    assert env.done
    env.trajectory_random(backward=True)
    assert env.is_source()


@pytest.mark.parametrize(
    "env, states, states_policy_exp",
    [
        (
            "env_grid2d_tetrismini",
            [
                [
                    [-1, 0, [0, 0], [0, 1]],
                    {
                        0: [0, 0],
                        1: torch.tensor(
                            [
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
                [
                    [0, 1, [0, 0], [0, 1]],
                    {
                        0: [1, 2],
                        1: torch.tensor(
                            [
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [400, 400, 000, 000],
                                [400, 400, 000, 000],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
                [
                    [1, 0, [1, 0], [0, 1]],
                    {
                        0: [1, 2],
                        1: torch.tensor(
                            [
                                [000, 000, 000, 000],
                                [000, 000, 100, 000],
                                [000, 000, 100, 000],
                                [400, 400, 100, 000],
                                [400, 400, 100, 000],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
            ],
            torch.stack(
                [
                    torch.tensor(
                        [
                            # fmt: off
                            0.0, 0.0, # ACTIVE SUBENV
                            0.0, # TOGGLE FLAG
                            0.0, 0.0, # DONES
                            # GRID
                            1.0, 0.0, 0.0,
                            1.0, 0.0, 0.0,
                            # TETRIS
                            0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0,
                            # fmt: on
                        ],
                        dtype=torch.float,
                    ),
                    torch.tensor(
                        [
                            # fmt: off
                            1.0, 0.0, # ACTIVE SUBENV
                            1.0, # TOGGLE FLAG
                            0.0, 0.0, # DONES
                            # GRID
                            0.0, 1.0, 0.0,
                            0.0, 0.0, 1.0,
                            # TETRIS
                            0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0,
                            1.0, 1.0, 0.0, 0.0,
                            1.0, 1.0, 0.0, 0.0,
                            # fmt: on
                        ],
                        dtype=torch.float,
                    ),
                    torch.tensor(
                        [
                            # fmt: off
                            0.0, 1.0, # ACTIVE SUBENV
                            0.0, # TOGGLE FLAG
                            1.0, 0.0, # DONES
                            # GRID
                            0.0, 1.0, 0.0,
                            0.0, 0.0, 1.0,
                            # TETRIS
                            0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 1.0, 0.0,
                            0.0, 0.0, 1.0, 0.0,
                            1.0, 1.0, 1.0, 0.0,
                            1.0, 1.0, 1.0, 0.0,
                            # fmt: on
                        ],
                        dtype=torch.float,
                    ),
                ],
                dim=0,
            ),
        ),
    ],
)
def test__states2policy__returns_expected(env, states, states_policy_exp, request):
    env = request.getfixturevalue(env)
    assert torch.equal(states_policy_exp, env.states2policy(states))


@pytest.mark.parametrize(
    "env, states, states_proxy_exp",
    [
        (
            "env_grid2d_tetrismini",
            [
                [
                    [-1, 0, [0, 0], [0, 1]],
                    {
                        0: [0, 0],
                        1: torch.tensor(
                            [
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
                [
                    [0, 1, [0, 0], [0, 1]],
                    {
                        0: [1, 2],
                        1: torch.tensor(
                            [
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [000, 000, 000, 000],
                                [400, 400, 000, 000],
                                [400, 400, 000, 000],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
                [
                    [1, 0, [1, 0], [0, 1]],
                    {
                        0: [1, 2],
                        1: torch.tensor(
                            [
                                [000, 000, 000, 000],
                                [000, 000, 100, 000],
                                [000, 000, 100, 000],
                                [400, 400, 100, 000],
                                [400, 400, 100, 000],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
            ],
            [
                [
                    [-1, 0, [0, 0], [0, 1]],
                    {
                        0: torch.tensor([-1.0, -1.0], dtype=torch.float),
                        1: torch.tensor(
                            [
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
                [
                    [0, 1, [0, 0], [0, 1]],
                    {
                        0: torch.tensor([0.0, 1.0], dtype=torch.float),
                        1: torch.tensor(
                            [
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [1, 1, 0, 0],
                                [1, 1, 0, 0],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
                [
                    [1, 0, [1, 0], [0, 1]],
                    {
                        0: torch.tensor([0.0, 1.0], dtype=torch.float),
                        1: torch.tensor(
                            [
                                [0, 0, 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 1, 0],
                                [1, 1, 1, 0],
                                [1, 1, 1, 0],
                            ],
                            dtype=torch.int16,
                        ),
                    },
                ],
            ],
        ),
    ],
)
def test__states2proxy__returns_expected(env, states, states_proxy_exp, request):
    env = request.getfixturevalue(env)
    for state, state_proxy_exp in zip(states, states_proxy_exp):
        assert env.equal(state_proxy_exp, env.state2proxy(state)[0])


class TestSetFixGrid2DTetrisMini(common.BaseTestsDiscrete):
    """Common tests for set of: Grid 3x3, Tetris-mini."""

    @pytest.fixture(autouse=True)
    def setup(self, env_grid2d_tetrismini):
        self.env = env_grid2d_tetrismini
        self.repeats = {
            "test__reset__state_is_source": 10,
            "test__forward_actions_have_nonzero_backward_prob": 10,
            "test__backward_actions_have_nonzero_forward_prob": 10,
            "test__trajectories_are_reversible": 10,
            "test__step_random__does_not_sample_invalid_actions_forward": 10,
            "test__step_random__does_not_sample_invalid_actions_backward": 10,
            "test__get_mask__is_consistent_regardless_of_inputs": 10,
            "test__get_valid_actions__is_consistent_regardless_of_inputs": 10,
            "test__sample_actions__get_logprobs__return_valid_actions_and_logprobs": 10,
            "test__get_parents_step_get_mask__are_compatible": 10,
            "test__sample_backwards_reaches_source": 10,
            "test__state2readable__is_reversible": 20,
            "test__gflownet_minimal_runs": 3,
        }
        self.n_states = {
            "test__backward_actions_have_nonzero_forward_prob": 3,
            "test__sample_backwards_reaches_source": 3,
            "test__get_logprobs__all_finite_in_random_forward_transitions": 10,
            "test__get_logprobs__all_finite_in_random_backward_transitions": 10,
        }
        self.batch_size = {
            "test__sample_actions__get_logprobs__batched_forward_trajectories": 10,
            "test__sample_actions__get_logprobs__batched_backward_trajectories": 10,
            "test__get_logprobs__all_finite_in_accumulated_forward_trajectories": 10,
            "test__gflownet_minimal_runs": 10,
        }


class TestSetFixCubeTetris(common.BaseTestsContinuous):
    """Common tests for set of: Cube, Tetris."""

    @pytest.fixture(autouse=True)
    def setup(self, env_cube_tetris):
        self.env = env_cube_tetris
        self.repeats = {
            "test__reset__state_is_source": 10,
            "test__forward_actions_have_nonzero_backward_prob": 10,
            "test__backward_actions_have_nonzero_forward_prob": 10,
            "test__trajectories_are_reversible": 10,
            "test__step_random__does_not_sample_invalid_actions_forward": 10,
            "test__step_random__does_not_sample_invalid_actions_backward": 10,
            "test__get_mask__is_consistent_regardless_of_inputs": 10,
            "test__get_valid_actions__is_consistent_regardless_of_inputs": 10,
            "test__sample_actions__get_logprobs__return_valid_actions_and_logprobs": 10,
            "test__get_parents_step_get_mask__are_compatible": 10,
            "test__sample_backwards_reaches_source": 10,
            "test__state2readable__is_reversible": 20,
            "test__gflownet_minimal_runs": 3,
        }
        self.n_states = {
            "test__backward_actions_have_nonzero_forward_prob": 3,
            "test__sample_backwards_reaches_source": 3,
            "test__get_logprobs__all_finite_in_random_forward_transitions": 10,
            "test__get_logprobs__all_finite_in_random_backward_transitions": 10,
        }
        self.batch_size = {
            "test__sample_actions__get_logprobs__batched_forward_trajectories": 10,
            "test__sample_actions__get_logprobs__batched_backward_trajectories": 10,
            "test__get_logprobs__all_finite_in_accumulated_forward_trajectories": 10,
            "test__gflownet_minimal_runs": 10,
        }


class TestSetFixCubeTetrisGrid(common.BaseTestsContinuous):
    """Common tests for set of: Grid 3x3, Tetris-mini."""

    @pytest.fixture(autouse=True)
    def setup(self, env_cube_tetris_grid):
        self.env = env_cube_tetris_grid
        self.repeats = {
            "test__reset__state_is_source": 10,
            "test__forward_actions_have_nonzero_backward_prob": 10,
            "test__backward_actions_have_nonzero_forward_prob": 10,
            "test__trajectories_are_reversible": 10,
            "test__step_random__does_not_sample_invalid_actions_forward": 10,
            "test__step_random__does_not_sample_invalid_actions_backward": 10,
            "test__get_mask__is_consistent_regardless_of_inputs": 10,
            "test__get_valid_actions__is_consistent_regardless_of_inputs": 10,
            "test__sample_actions__get_logprobs__return_valid_actions_and_logprobs": 10,
            "test__get_parents_step_get_mask__are_compatible": 10,
            "test__sample_backwards_reaches_source": 10,
            "test__state2readable__is_reversible": 20,
            "test__gflownet_minimal_runs": 3,
        }
        self.n_states = {
            "test__backward_actions_have_nonzero_forward_prob": 3,
            "test__sample_backwards_reaches_source": 3,
            "test__get_logprobs__all_finite_in_random_forward_transitions": 10,
            "test__get_logprobs__all_finite_in_random_backward_transitions": 10,
        }
        self.batch_size = {
            "test__sample_actions__get_logprobs__batched_forward_trajectories": 10,
            "test__sample_actions__get_logprobs__batched_backward_trajectories": 10,
            "test__get_logprobs__all_finite_in_accumulated_forward_trajectories": 10,
            "test__gflownet_minimal_runs": 10,
        }


class TestSetFixTwoGrids(common.BaseTestsDiscrete):
    """Common tests for set of two Grids."""

    @pytest.fixture(autouse=True)
    def setup(self, env_two_grids):
        self.env = env_two_grids
        self.repeats = {
            "test__reset__state_is_source": 10,
            "test__forward_actions_have_nonzero_backward_prob": 10,
            "test__backward_actions_have_nonzero_forward_prob": 10,
            "test__trajectories_are_reversible": 10,
            "test__step_random__does_not_sample_invalid_actions_forward": 10,
            "test__step_random__does_not_sample_invalid_actions_backward": 10,
            "test__get_mask__is_consistent_regardless_of_inputs": 10,
            "test__get_valid_actions__is_consistent_regardless_of_inputs": 10,
            "test__sample_actions__get_logprobs__return_valid_actions_and_logprobs": 10,
            "test__get_parents_step_get_mask__are_compatible": 10,
            "test__sample_backwards_reaches_source": 10,
            "test__state2readable__is_reversible": 20,
            "test__gflownet_minimal_runs": 3,
        }
        self.n_states = {
            "test__backward_actions_have_nonzero_forward_prob": 3,
            "test__sample_backwards_reaches_source": 3,
            "test__get_logprobs__all_finite_in_random_forward_transitions": 10,
            "test__get_logprobs__all_finite_in_random_backward_transitions": 10,
        }
        self.batch_size = {
            "test__sample_actions__get_logprobs__batched_forward_trajectories": 10,
            "test__sample_actions__get_logprobs__batched_backward_trajectories": 10,
            "test__get_logprobs__all_finite_in_accumulated_forward_trajectories": 10,
            "test__gflownet_minimal_runs": 10,
        }


class TestSetFixThreeCubes(common.BaseTestsContinuous):
    """Common tests for set of three cubes."""

    @pytest.fixture(autouse=True)
    def setup(self, env_three_cubes):
        self.env = env_three_cubes
        self.repeats = {
            "test__reset__state_is_source": 10,
            "test__forward_actions_have_nonzero_backward_prob": 10,
            "test__backward_actions_have_nonzero_forward_prob": 10,
            "test__trajectories_are_reversible": 10,
            "test__step_random__does_not_sample_invalid_actions_forward": 10,
            "test__step_random__does_not_sample_invalid_actions_backward": 10,
            "test__get_mask__is_consistent_regardless_of_inputs": 10,
            "test__get_valid_actions__is_consistent_regardless_of_inputs": 10,
            "test__sample_actions__get_logprobs__return_valid_actions_and_logprobs": 10,
            "test__get_parents_step_get_mask__are_compatible": 10,
            "test__sample_backwards_reaches_source": 10,
            "test__state2readable__is_reversible": 20,
            "test__gflownet_minimal_runs": 3,
        }
        self.n_states = {
            "test__backward_actions_have_nonzero_forward_prob": 3,
            "test__sample_backwards_reaches_source": 3,
            "test__get_logprobs__all_finite_in_random_forward_transitions": 10,
            "test__get_logprobs__all_finite_in_random_backward_transitions": 10,
        }
        self.batch_size = {
            "test__sample_actions__get_logprobs__batched_forward_trajectories": 10,
            "test__sample_actions__get_logprobs__batched_backward_trajectories": 10,
            "test__get_logprobs__all_finite_in_accumulated_forward_trajectories": 10,
            "test__gflownet_minimal_runs": 10,
        }


class TestSetFixCube2DCube3D(common.BaseTestsContinuous):
    """Common tests for set of three cubes."""

    @pytest.fixture(autouse=True)
    def setup(self, env_cube2d_cube3d):
        self.env = env_cube2d_cube3d
        self.repeats = {
            "test__reset__state_is_source": 10,
            "test__forward_actions_have_nonzero_backward_prob": 10,
            "test__backward_actions_have_nonzero_forward_prob": 10,
            "test__trajectories_are_reversible": 10,
            "test__step_random__does_not_sample_invalid_actions_forward": 10,
            "test__step_random__does_not_sample_invalid_actions_backward": 10,
            "test__get_mask__is_consistent_regardless_of_inputs": 10,
            "test__get_valid_actions__is_consistent_regardless_of_inputs": 10,
            "test__sample_actions__get_logprobs__return_valid_actions_and_logprobs": 10,
            "test__get_parents_step_get_mask__are_compatible": 10,
            "test__sample_backwards_reaches_source": 10,
            "test__state2readable__is_reversible": 20,
            "test__gflownet_minimal_runs": 3,
        }
        self.n_states = {
            "test__backward_actions_have_nonzero_forward_prob": 3,
            "test__sample_backwards_reaches_source": 3,
            "test__get_logprobs__all_finite_in_random_forward_transitions": 10,
            "test__get_logprobs__all_finite_in_random_backward_transitions": 10,
        }
        self.batch_size = {
            "test__sample_actions__get_logprobs__batched_forward_trajectories": 10,
            "test__sample_actions__get_logprobs__batched_backward_trajectories": 10,
            "test__get_logprobs__all_finite_in_accumulated_forward_trajectories": 10,
            "test__gflownet_minimal_runs": 10,
        }


class TestSetFixTwoCubes2DOneCube3D(common.BaseTestsContinuous):
    """Common tests for set of three cubes."""

    @pytest.fixture(autouse=True)
    def setup(self, env_two_cubes2d_one_cube3d):
        self.env = env_two_cubes2d_one_cube3d
        self.repeats = {
            "test__reset__state_is_source": 10,
            "test__forward_actions_have_nonzero_backward_prob": 10,
            "test__backward_actions_have_nonzero_forward_prob": 10,
            "test__trajectories_are_reversible": 10,
            "test__step_random__does_not_sample_invalid_actions_forward": 10,
            "test__step_random__does_not_sample_invalid_actions_backward": 10,
            "test__get_mask__is_consistent_regardless_of_inputs": 10,
            "test__get_valid_actions__is_consistent_regardless_of_inputs": 10,
            "test__sample_actions__get_logprobs__return_valid_actions_and_logprobs": 10,
            "test__get_parents_step_get_mask__are_compatible": 10,
            "test__sample_backwards_reaches_source": 10,
            "test__state2readable__is_reversible": 20,
            "test__gflownet_minimal_runs": 3,
        }
        self.n_states = {
            "test__backward_actions_have_nonzero_forward_prob": 3,
            "test__sample_backwards_reaches_source": 3,
            "test__get_logprobs__all_finite_in_random_forward_transitions": 10,
            "test__get_logprobs__all_finite_in_random_backward_transitions": 10,
        }
        self.batch_size = {
            "test__sample_actions__get_logprobs__batched_forward_trajectories": 10,
            "test__sample_actions__get_logprobs__batched_backward_trajectories": 10,
            "test__get_logprobs__all_finite_in_accumulated_forward_trajectories": 10,
            "test__gflownet_minimal_runs": 10,
        }


class TestSetFixStacksEqual(common.BaseTestsContinuous):
    """Common tests for set of three cubes."""

    @pytest.fixture(autouse=True)
    def setup(self, env_stacks_equal):
        self.env = env_stacks_equal
        self.repeats = {
            "test__reset__state_is_source": 10,
            "test__forward_actions_have_nonzero_backward_prob": 10,
            "test__backward_actions_have_nonzero_forward_prob": 10,
            "test__trajectories_are_reversible": 10,
            "test__step_random__does_not_sample_invalid_actions_forward": 10,
            "test__step_random__does_not_sample_invalid_actions_backward": 10,
            "test__get_mask__is_consistent_regardless_of_inputs": 10,
            "test__get_valid_actions__is_consistent_regardless_of_inputs": 10,
            "test__sample_actions__get_logprobs__return_valid_actions_and_logprobs": 10,
            "test__get_parents_step_get_mask__are_compatible": 10,
            "test__sample_backwards_reaches_source": 10,
            "test__state2readable__is_reversible": 20,
            "test__gflownet_minimal_runs": 3,
        }
        self.n_states = {
            "test__backward_actions_have_nonzero_forward_prob": 3,
            "test__sample_backwards_reaches_source": 3,
            "test__get_logprobs__all_finite_in_random_forward_transitions": 10,
            "test__get_logprobs__all_finite_in_random_backward_transitions": 10,
        }
        self.batch_size = {
            "test__sample_actions__get_logprobs__batched_forward_trajectories": 10,
            "test__sample_actions__get_logprobs__batched_backward_trajectories": 10,
            "test__get_logprobs__all_finite_in_accumulated_forward_trajectories": 10,
            "test__gflownet_minimal_runs": 10,
        }


class TestSetFixStacksDiff(common.BaseTestsContinuous):
    """Common tests for set of three cubes."""

    @pytest.fixture(autouse=True)
    def setup(self, env_stacks_diff):
        self.env = env_stacks_diff
        self.repeats = {
            "test__reset__state_is_source": 10,
            "test__forward_actions_have_nonzero_backward_prob": 10,
            "test__backward_actions_have_nonzero_forward_prob": 10,
            "test__trajectories_are_reversible": 10,
            "test__step_random__does_not_sample_invalid_actions_forward": 10,
            "test__step_random__does_not_sample_invalid_actions_backward": 10,
            "test__get_mask__is_consistent_regardless_of_inputs": 10,
            "test__get_valid_actions__is_consistent_regardless_of_inputs": 10,
            "test__sample_actions__get_logprobs__return_valid_actions_and_logprobs": 10,
            "test__get_parents_step_get_mask__are_compatible": 10,
            "test__sample_backwards_reaches_source": 10,
            "test__state2readable__is_reversible": 20,
            "test__gflownet_minimal_runs": 3,
        }
        self.n_states = {
            "test__backward_actions_have_nonzero_forward_prob": 3,
            "test__sample_backwards_reaches_source": 3,
            "test__get_logprobs__all_finite_in_random_forward_transitions": 10,
            "test__get_logprobs__all_finite_in_random_backward_transitions": 10,
        }
        self.batch_size = {
            "test__sample_actions__get_logprobs__batched_forward_trajectories": 10,
            "test__sample_actions__get_logprobs__batched_backward_trajectories": 10,
            "test__get_logprobs__all_finite_in_accumulated_forward_trajectories": 10,
            "test__gflownet_minimal_runs": 10,
        }
