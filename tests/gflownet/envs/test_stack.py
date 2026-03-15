import warnings

import common
import numpy as np
import pytest
import torch
from torch import Tensor

from gflownet.envs.composite.setfix import SetFix
from gflownet.envs.composite.stack import Stack
from gflownet.envs.cube import ContinuousCube
from gflownet.envs.grid import Grid
from gflownet.envs.tetris import Tetris
from gflownet.utils.common import tbool, tfloat


@pytest.fixture
def env_grid2d_tetrismini():
    return Stack(
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
    return Stack(
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
    return Stack(
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
def env_cube_setgrids():
    return Stack(
        subenvs=(
            ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
            SetFix(
                subenvs=(
                    Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),
                    Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),
                )
            ),
        )
    )


@pytest.fixture
def env_cube_setstacks():
    """
    Stack of set of stacks, to test compositionality of meta environments.
    0: Cube
    1: SetFix
        Stack
            0: Cube
            1: Grid
        Stack
            0: Grid
            1: Cube
    """
    return Stack(
        subenvs=(
            ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
            SetFix(
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
            ),
        )
    )


@pytest.mark.parametrize(
    "env",
    [
        "env_grid2d_tetrismini",
        "env_cube_tetris",
        "env_cube_tetris_grid",
        "env_cube_setgrids",
        "env_cube_setstacks",
    ],
)
def test__environment__initializes_properly(env, request):
    env = request.getfixturevalue(env)
    assert True


@pytest.mark.parametrize(
    "env, is_continuous",
    [
        ("env_grid2d_tetrismini", False),
        ("env_cube_tetris", True),
        ("env_cube_tetris_grid", True),
    ],
)
def test__environment__is_continuous(env, is_continuous, request):
    env = request.getfixturevalue(env)
    assert env.continuous == is_continuous


@pytest.mark.parametrize(
    "env, action, representative",
    [
        ("env_grid2d_tetrismini", (0, 1, 0, 0), (0, 1, 0, 0)),
        ("env_grid2d_tetrismini", (1, 1, 0, 3), (1, 1, 0, 3)),
        ("env_cube_tetris", (0, 0.2, 0.3, 0), (0, 0, 0, 0)),
        ("env_cube_tetris", (0, 0.5, 0.7, 1), (0, 0, 0, 1)),
        ("env_cube_tetris", (1, 4, 0, 2), (1, 4, 0, 2)),
        ("env_cube_tetris_grid", (0, 0.5, 0.7, 1), (0, 0, 0, 1)),
        ("env_cube_tetris_grid", (1, 4, 0, 2), (1, 4, 0, 2)),
        ("env_cube_tetris_grid", (2, 0, 1, 0), (2, 0, 1, 0)),
        ("env_cube_tetris_grid", (2, 0, 0, 0), (2, 0, 0, 0)),
    ],
)
def test__action2representative(env, action, representative, request):
    env = request.getfixturevalue(env)
    assert env.action2representative(action) == representative


@pytest.mark.parametrize(
    "env, action_stack, action_subenv",
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
    ],
)
def test__pad_depad_action__return_expected(env, action_stack, action_subenv, request):
    env = request.getfixturevalue(env)
    stage = action_stack[0]
    # Check pad
    assert env._pad_action(action_subenv, stage) == action_stack
    # Check depad
    assert env._depad_action(action_stack, stage) == action_subenv


@pytest.mark.parametrize(
    "action_space",
    [
        [
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
        ]
    ],
)
def test__get_action_space__returns_expected(env_grid2d_tetrismini, action_space):
    env = env_grid2d_tetrismini
    assert set(action_space) == set(env.action_space)


@pytest.mark.parametrize(
    "env", ["env_grid2d_tetrismini", "env_cube_tetris", "env_cube_tetris_grid"]
)
def test__action_space__contains_actions_of_all_unique_envs(env, request):
    env = request.getfixturevalue(env)
    action_space = env.action_space
    for idx_unique, env_unique in zip(env.unique_indices, env.envs_unique):
        assert all(
            [
                env._pad_action(action, idx_unique) in action_space
                for action in env_unique.action_space
            ]
        )


@pytest.mark.parametrize(
    "env, mask_dim_expected",
    [
        ("env_grid2d_tetrismini", max([3, 8]) + 2),
        ("env_cube_tetris", max([5, 6]) + 2),
        ("env_cube_tetris_grid", max([5, 8, 3]) + 3),
    ],
)
def test__mask_dim__is_as_expected(env, request, mask_dim_expected):
    env = request.getfixturevalue(env)
    assert env.mask_dim == mask_dim_expected


@pytest.mark.parametrize(
    "env, source",
    [
        (
            "env_grid2d_tetrismini",
            {
                "_active": 0,
                "_envs_unique": [0, 1],
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
                    device="cpu",
                ),
            },
        ),
        (
            "env_cube_tetris",
            {
                "_active": 0,
                "_envs_unique": [0, 1],
                0: [-1.0, -1.0],
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
                    device="cpu",
                ),
            },
        ),
        (
            "env_cube_tetris_grid",
            {
                "_active": 0,
                "_envs_unique": [0, 1, 2],
                0: [-1.0, -1.0],
                1: torch.tensor(
                    [
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                    ],
                    dtype=torch.int16,
                    device="cpu",
                ),
                2: [0, 0, 0],
            },
        ),
    ],
)
def test__source_is_expected(env, source, request):
    env = request.getfixturevalue(env)
    assert env.equal(env.source, source)


@pytest.mark.parametrize(
    "env, state, is_source",
    [
        (
            "env_grid2d_tetrismini",
            {
                "_active": 0,
                "_envs_unique": [0, 1],
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
                    device="cpu",
                ),
            },
            True,
        ),
        (
            "env_cube_tetris",
            {
                "_active": 0,
                "_envs_unique": [0, 1],
                0: [-1.0, -1.0],
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
                    device="cpu",
                ),
            },
            True,
        ),
        (
            "env_cube_tetris_grid",
            {
                "_active": 0,
                "_envs_unique": [0, 1, 2],
                0: [-1.0, -1.0],
                1: torch.tensor(
                    [
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                    ],
                    dtype=torch.int16,
                    device="cpu",
                ),
                2: [0, 0, 0],
            },
            True,
        ),
        (
            "env_cube_tetris_grid",
            {
                "_active": 1,
                "_envs_unique": [0, 1, 2],
                0: [-1.0, -1.0],
                1: torch.tensor(
                    [
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                    ],
                    dtype=torch.int16,
                    device="cpu",
                ),
                2: [0, 0, 0],
            },
            False,
        ),
        (
            "env_cube_setgrids",
            {
                "_active": 0,
                "_envs_unique": [0, 1],
                0: [-1, -1],
                1: {
                    "_active": -1,
                    "_toggle": 0,
                    "_dones": [0, 0],
                    "_envs_unique": [0, 0],
                    "_keys": [0, 1],
                    0: [0, 0],
                    1: [0, 0],
                },
            },
            True,
        ),
        (
            "env_cube_setgrids",
            {
                "_active": 1,
                "_envs_unique": [0, 1],
                0: [-1, -1],
                1: {
                    "_active": -1,
                    "_toggle": 0,
                    "_dones": [0, 0],
                    "_envs_unique": [0, 0],
                    "_keys": [0, 1],
                    0: [0, 0],
                    1: [0, 0],
                },
            },
            False,
        ),
        (
            "env_cube_setgrids",
            {
                "_active": 0,
                "_envs_unique": [0, 1],
                0: [0.0, 0.0],
                1: {
                    "_active": -1,
                    "_toggle": 0,
                    "_dones": [0, 0],
                    "_envs_unique": [0, 0],
                    "_keys": [0, 1],
                    0: [0, 0],
                    1: [0, 0],
                },
            },
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
    "env, state, dones",
    [
        (
            "env_grid2d_tetrismini",
            {
                "_active": 0,
                "_envs_unique": [0, 1],
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
                    device="cpu",
                ),
            },
            [False, False],
        ),
        (
            "env_grid2d_tetrismini",
            {
                "_active": 1,
                "_envs_unique": [0, 1],
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
                    device="cpu",
                ),
            },
            [True, False],
        ),
        (
            "env_grid2d_tetrismini",
            {
                "_active": 0,
                "_envs_unique": [0, 1],
                0: [1, 2],
                1: torch.tensor(
                    [
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                    ],
                    dtype=torch.int16,
                    device="cpu",
                ),
            },
            [False, False],
        ),
        (
            "env_grid2d_tetrismini",
            {
                "_active": 1,
                "_envs_unique": [0, 1],
                0: [1, 2],
                1: torch.tensor(
                    [
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                    ],
                    dtype=torch.int16,
                    device="cpu",
                ),
            },
            [True, False],
        ),
        (
            "env_grid2d_tetrismini",
            {
                "_active": 1,
                "_envs_unique": [0, 1],
                0: [1, 2],
                1: torch.tensor(
                    [
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                        [000, 400, 400, 000],
                        [000, 400, 400, 000],
                    ],
                    dtype=torch.int16,
                    device="cpu",
                ),
            },
            [True, False],
        ),
        (
            "env_grid2d_tetrismini",
            {
                "_active": 1,
                "_envs_unique": [0, 1],
                0: [1, 2],
                1: torch.tensor(
                    [
                        [000, 000, 000, 000],
                        [100, 000, 000, 000],
                        [100, 000, 000, 000],
                        [100, 400, 400, 000],
                        [100, 400, 400, 000],
                    ],
                    dtype=torch.int16,
                    device="cpu",
                ),
            },
            [True, False],
        ),
        (
            "env_grid2d_tetrismini",
            {
                "_active": 1,
                "_envs_unique": [0, 1],
                0: [1, 2],
                1: torch.tensor(
                    [
                        [000, 000, 000, 000],
                        [100, 000, 401, 401],
                        [100, 000, 401, 401],
                        [100, 400, 400, 000],
                        [100, 400, 400, 000],
                    ],
                    dtype=torch.int16,
                    device="cpu",
                ),
            },
            [True, True],
        ),
        (
            "env_cube_setgrids",
            {
                "_active": 0,
                "_envs_unique": [0, 1],
                0: [-1, -1],
                1: {
                    "_active": -1,
                    "_toggle": 0,
                    "_dones": [0, 0],
                    "_envs_unique": [0, 0],
                    "_keys": [0, 1],
                    0: [0, 0],
                    1: [0, 0],
                },
            },
            [False, False],
        ),
        (
            "env_cube_setgrids",
            {
                "_active": 0,
                "_envs_unique": [0, 1],
                0: [0.3, 0.7],
                1: {
                    "_active": -1,
                    "_toggle": 0,
                    "_dones": [0, 0],
                    "_envs_unique": [0, 0],
                    "_keys": [0, 1],
                    0: [0, 0],
                    1: [0, 0],
                },
            },
            [False, False],
        ),
        (
            "env_cube_setgrids",
            {
                "_active": 1,
                "_envs_unique": [0, 1],
                0: [0.3, 0.7],
                1: {
                    "_active": -1,
                    "_toggle": 0,
                    "_dones": [0, 0],
                    "_envs_unique": [0, 0],
                    "_keys": [0, 1],
                    0: [0, 0],
                    1: [0, 0],
                },
            },
            [True, False],
        ),
        (
            "env_cube_setgrids",
            {
                "_active": 1,
                "_envs_unique": [0, 1],
                0: [0.3, 0.7],
                1: {
                    "_active": 0,
                    "_toggle": 0,
                    "_dones": [0, 0],
                    "_envs_unique": [0, 0],
                    "_keys": [0, 1],
                    0: [1, 0],
                    1: [0, 0],
                },
            },
            [True, False],
        ),
        (
            "env_cube_setgrids",
            {
                "_active": 1,
                "_envs_unique": [0, 1],
                0: [0.3, 0.7],
                1: {
                    "_active": -1,
                    "_toggle": 0,
                    "_dones": [1, 1],
                    "_envs_unique": [0, 0],
                    "_keys": [0, 1],
                    0: [1, 2],
                    1: [1, 1],
                },
            },
            [True, True],
        ),
        (
            "env_cube_setstacks",
            {
                "_active": 0,
                "_envs_unique": [0, 1],
                0: [-1, -1],
                1: {
                    "_active": -1,
                    "_toggle": 0,
                    "_dones": [0, 0],
                    "_envs_unique": [0, 1],
                    "_keys": [0, 1],
                    0: {
                        "_active": 0,
                        "_envs_unique": [0, 1],
                        0: [-1.0, -1.0],
                        1: [0, 0],
                    },
                    1: {
                        "_active": 0,
                        "_envs_unique": [0, 1],
                        0: [0, 0],
                        1: [-1.0, -1.0],
                    },
                },
            },
            [False, False],
        ),
        (
            "env_cube_setstacks",
            {
                "_active": 0,
                "_envs_unique": [0, 1],
                0: [0.3, 0.7],
                1: {
                    "_active": -1,
                    "_toggle": 0,
                    "_dones": [0, 0],
                    "_envs_unique": [0, 1],
                    "_keys": [0, 1],
                    0: {
                        "_active": 0,
                        "_envs_unique": [0, 1],
                        0: [-1.0, -1.0],
                        1: [0, 0],
                    },
                    1: {
                        "_active": 0,
                        "_envs_unique": [0, 1],
                        0: [0, 0],
                        1: [-1.0, -1.0],
                    },
                },
            },
            [False, False],
        ),
        (
            "env_cube_setstacks",
            {
                "_active": 1,
                "_envs_unique": [0, 1],
                0: [0.3, 0.7],
                1: {
                    "_active": -1,
                    "_toggle": 0,
                    "_dones": [0, 0],
                    "_envs_unique": [0, 1],
                    "_keys": [0, 1],
                    0: {
                        "_active": 0,
                        "_envs_unique": [0, 1],
                        0: [-1.0, -1.0],
                        1: [0, 0],
                    },
                    1: {
                        "_active": 0,
                        "_envs_unique": [0, 1],
                        0: [0, 0],
                        1: [-1.0, -1.0],
                    },
                },
            },
            [True, False],
        ),
        (
            "env_cube_setstacks",
            {
                "_active": 1,
                "_envs_unique": [0, 1],
                0: [0.3, 0.7],
                1: {
                    "_active": 0,
                    "_toggle": 0,
                    "_dones": [0, 0],
                    "_envs_unique": [0, 1],
                    "_keys": [0, 1],
                    0: {"_active": 0, "_envs_unique": [0, 1], 0: [0.1, 0.2], 1: [0, 0]},
                    1: {
                        "_active": 0,
                        "_envs_unique": [0, 1],
                        0: [0, 0],
                        1: [-1.0, -1.0],
                    },
                },
            },
            [True, False],
        ),
        (
            "env_cube_setstacks",
            {
                "_active": 1,
                "_envs_unique": [0, 1],
                0: [0.3, 0.7],
                1: {
                    "_active": -1,
                    "_toggle": 0,
                    "_dones": [1, 0],
                    "_envs_unique": [0, 1],
                    "_keys": [0, 1],
                    0: {"_active": 1, "_envs_unique": [0, 1], 0: [0.1, 0.2], 1: [2, 1]},
                    1: {
                        "_active": 0,
                        "_envs_unique": [0, 1],
                        0: [0, 0],
                        1: [-1.0, -1.0],
                    },
                },
            },
            [True, False],
        ),
        (
            "env_cube_setstacks",
            {
                "_active": 1,
                "_envs_unique": [0, 1],
                0: [0.3, 0.7],
                1: {
                    "_active": -1,
                    "_toggle": 0,
                    "_dones": [1, 1],
                    "_envs_unique": [0, 1],
                    "_keys": [0, 1],
                    0: {"_active": 1, "_envs_unique": [0, 1], 0: [0.1, 0.2], 1: [2, 1]},
                    1: {"_active": 0, "_envs_unique": [0, 1], 0: [1, 2], 1: [0.3, 0.8]},
                },
            },
            [True, True],
        ),
    ],
)
def test__set_state__sets_state_and_dones(env, state, dones, request):
    env = request.getfixturevalue(env)
    if all(dones):
        env.set_state(state, done=True)
    else:
        env.set_state(state, done=False)

    # Check global state
    assert env.equal(env.state, state)

    # Check states of subenvs
    for idx, subenv in enumerate(env.subenvs):
        assert env.equal(subenv.state, env._get_substate(state, idx))

    # Check dones
    for subenv, done in zip(env.subenvs, dones):
        assert subenv.done == done


@pytest.mark.parametrize(
    "env, state",
    [
        (
            "env_grid2d_tetrismini",
            {
                "_active": 0,
                "_envs_unique": [0, 1],
                0: [2, 0],
                1: torch.tensor(
                    [
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                    ],
                    dtype=torch.int16,
                    device="cpu",
                ),
            },
        ),
        (
            "env_grid2d_tetrismini",
            {
                "_active": 1,
                "_envs_unique": [0, 1],
                0: [2, 0],
                1: torch.tensor(
                    [
                        [000, 000, 000, 000],
                        [100, 000, 000, 000],
                        [100, 000, 000, 000],
                        [100, 000, 000, 000],
                        [100, 000, 000, 000],
                    ],
                    dtype=torch.int16,
                    device="cpu",
                ),
            },
        ),
        (
            "env_cube_tetris",
            {
                "_active": 0,
                "_envs_unique": [0, 1],
                0: [0.3, 0.7],
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
                    device="cpu",
                ),
            },
        ),
        (
            "env_cube_tetris",
            {
                "_active": 1,
                "_envs_unique": [0, 1],
                0: [0.3, 0.7],
                1: torch.tensor(
                    [
                        [000, 200],
                        [000, 200],
                        [200, 200],
                        [300, 000],
                        [300, 000],
                        [300, 300],
                    ],
                    dtype=torch.int16,
                    device="cpu",
                ),
            },
        ),
        (
            "env_cube_tetris_grid",
            {
                "_active": 2,
                "_envs_unique": [0, 1, 2],
                0: [0.3, 0.7],
                1: torch.tensor(
                    [
                        [000, 200],
                        [000, 200],
                        [200, 200],
                        [300, 000],
                        [300, 000],
                        [300, 300],
                    ],
                    dtype=torch.int16,
                    device="cpu",
                ),
                2: [1, 0, 2],
            },
        ),
    ],
)
def test__get_mask_invalid_actions_backward__returns_expected_general_case(
    env, state, request
):
    env = request.getfixturevalue(env)
    active_subenv = env._get_active_subenv(state)
    subenv = env.subenvs[active_subenv]
    mask = env.get_mask_invalid_actions_backward(state, done=False)
    mask_subenv = env._unformat_mask(mask, active_subenv)
    mask_subenv_expected = subenv.get_mask_invalid_actions_backward(
        env._get_substate(state, active_subenv)
    )
    assert mask_subenv == mask_subenv_expected, state


@pytest.mark.parametrize(
    "env, state",
    [
        # Tetris source
        (
            "env_grid2d_tetrismini",
            {
                "_active": 1,
                "_envs_unique": [0, 1],
                0: [2, 0],
                1: torch.tensor(
                    [
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                    ],
                    dtype=torch.int16,
                    device="cpu",
                ),
            },
        ),
        (
            "env_cube_tetris",
            {
                "_active": 1,
                "_envs_unique": [0, 1],
                0: [0.3, 0.7],
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
                    device="cpu",
                ),
            },
        ),
        # Active subenv (2) is source
        (
            "env_cube_tetris_grid",
            {
                "_active": 2,
                "_envs_unique": [0, 1, 2],
                0: [0.3, 0.7],
                1: torch.tensor(
                    [
                        [000, 000, 000, 000],
                        [000, 000, 200, 000],
                        [300, 000, 200, 000],
                        [300, 200, 200, 000],
                        [300, 300, 000, 000],
                    ],
                    dtype=torch.int16,
                    device="cpu",
                ),
                2: [0, 0, 0],
            },
        ),
        # Active subenv (1) is source
        (
            "env_cube_tetris_grid",
            {
                "_active": 1,
                "_envs_unique": [0, 1, 2],
                0: [0.3, 0.7],
                1: torch.tensor(
                    [
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                    ],
                    dtype=torch.int16,
                    device="cpu",
                ),
                2: [0, 0, 0],
            },
        ),
    ],
)
def test__get_mask_invalid_actions_backward__returns_expected_stage_transition(
    env, state, request
):
    env = request.getfixturevalue(env)
    active_subenv = env._get_active_subenv(state)
    relevant_subenv = active_subenv - 1
    subenv = env.subenvs[relevant_subenv]
    state_subenv = env._get_substate(state, relevant_subenv)
    # Get the global mask and extract the relevant part
    mask = env.get_mask_invalid_actions_backward(state, done=False)
    mask_subenv = env._unformat_mask(mask, relevant_subenv)
    # Get expected mask of the relevant subenv
    mask_subenv_expected = subenv.get_mask_invalid_actions_backward(
        state_subenv, done=True
    )
    assert mask_subenv == mask_subenv_expected, state


@pytest.mark.parametrize(
    "env, state",
    [
        (
            "env_grid2d_tetrismini",
            {
                "_active": 1,
                "_envs_unique": [0, 1],
                0: [2, 0],
                1: torch.tensor(
                    [
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                        [300, 000, 000, 000],
                        [300, 000, 000, 000],
                        [300, 300, 000, 000],
                    ],
                    dtype=torch.int16,
                    device="cpu",
                ),
            },
        ),
        # Last substate is source (but done)
        (
            "env_grid2d_tetrismini",
            {
                "_active": 1,
                "_envs_unique": [0, 1],
                0: [2, 0],
                1: torch.tensor(
                    [
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                    ],
                    dtype=torch.int16,
                    device="cpu",
                ),
            },
        ),
        (
            "env_cube_tetris",
            {
                "_active": 1,
                "_envs_unique": [0, 1],
                0: [0.3, 0.7],
                1: torch.tensor(
                    [
                        [000, 000],
                        [000, 000],
                        [000, 000],
                        [300, 000],
                        [300, 000],
                        [300, 300],
                    ],
                    dtype=torch.int16,
                    device="cpu",
                ),
            },
        ),
        # Last substate is source (but done)
        (
            "env_cube_tetris",
            {
                "_active": 1,
                "_envs_unique": [0, 1],
                0: [0.3, 0.7],
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
                    device="cpu",
                ),
            },
        ),
        (
            "env_cube_tetris_grid",
            {
                "_active": 2,
                "_envs_unique": [0, 1, 2],
                0: [0.3, 0.7],
                1: torch.tensor(
                    [
                        [000, 000, 000, 000],
                        [000, 000, 200, 000],
                        [300, 000, 200, 000],
                        [300, 200, 200, 000],
                        [300, 300, 000, 000],
                    ],
                    dtype=torch.int16,
                    device="cpu",
                ),
                2: [1, 2, 0],
            },
        ),
        # Last substate is source (but done)
        (
            "env_cube_tetris_grid",
            {
                "_active": 2,
                "_envs_unique": [0, 1, 2],
                0: [0.3, 0.7],
                1: torch.tensor(
                    [
                        [000, 000, 000, 000],
                        [000, 000, 200, 000],
                        [300, 000, 200, 000],
                        [300, 200, 200, 000],
                        [300, 300, 000, 000],
                    ],
                    dtype=torch.int16,
                    device="cpu",
                ),
                2: [0, 0, 0],
            },
        ),
    ],
)
def test__get_mask_invalid_actions_backward__returns_expected_global_done(
    env, state, request
):
    env = request.getfixturevalue(env)
    active_subenv = env._get_active_subenv(state)
    relevant_subenv = active_subenv
    subenv = env.subenvs[relevant_subenv]
    state_subenv = env._get_substate(state, relevant_subenv)
    # Get the global mask and extract the relevant part
    mask = env.get_mask_invalid_actions_backward(state, done=True)
    mask_subenv = env._unformat_mask(mask, relevant_subenv)
    # Get expected mask of the relevant subenv
    mask_subenv_expected = subenv.get_mask_invalid_actions_backward(
        state_subenv, done=True
    )
    assert mask_subenv == mask_subenv_expected, state


@pytest.mark.parametrize(
    "env, state",
    [
        (
            "env_grid2d_tetrismini",
            {
                "_active": 0,
                "_envs_unique": [0, 1],
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
                    device="cpu",
                ),
            },
        ),
        (
            "env_cube_tetris",
            {
                "_active": 0,
                "_envs_unique": [0, 1],
                0: [-1.0, -1.0],
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
                    device="cpu",
                ),
            },
        ),
        (
            "env_cube_tetris_grid",
            {
                "_active": 0,
                "_envs_unique": [0, 1, 2],
                0: [-1.0, -1.0],
                1: torch.tensor(
                    [
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                    ],
                    dtype=torch.int16,
                    device="cpu",
                ),
                2: [0, 0, 0],
            },
        ),
    ],
)
def test__get_mask_invalid_actions_backward__returns_expected_global_source(
    env, state, request
):
    env = request.getfixturevalue(env)
    active_subenv = 0
    relevant_subenv = active_subenv
    subenv = env.subenvs[relevant_subenv]
    state_subenv = env._get_substate(state, relevant_subenv)
    # Get the global mask and extract the relevant part
    mask = env.get_mask_invalid_actions_backward(state, done=False)
    mask_subenv = env._unformat_mask(mask, relevant_subenv)
    # Get expected mask of the relevant subenv
    mask_subenv_expected = subenv.get_mask_invalid_actions_backward(
        state_subenv, done=False
    )
    assert mask_subenv == mask_subenv_expected, state


@pytest.mark.parametrize(
    "env, state_from, action, state_next_exp, valid_exp",
    [
        (
            "env_grid2d_tetrismini",
            {
                "_active": 0,
                "_envs_unique": [0, 1],
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
                    device="cpu",
                ),
            },
            (0, 1, 0, 0),
            {
                "_active": 0,
                "_envs_unique": [0, 1],
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
                    device="cpu",
                ),
            },
            True,
        ),
        (
            "env_grid2d_tetrismini",
            {
                "_active": 0,
                "_envs_unique": [0, 1],
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
                    device="cpu",
                ),
            },
            (0, 1, 0, 0),
            {
                "_active": 0,
                "_envs_unique": [0, 1],
                0: [2, 0],
                1: torch.tensor(
                    [
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                    ],
                    dtype=torch.int16,
                    device="cpu",
                ),
            },
            True,
        ),
        # EOS action from Grid must increment stage
        (
            "env_grid2d_tetrismini",
            {
                "_active": 0,
                "_envs_unique": [0, 1],
                0: [2, 0],
                1: torch.tensor(
                    [
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                    ],
                    dtype=torch.int16,
                    device="cpu",
                ),
            },
            (0, 0, 0, 0),
            {
                "_active": 1,
                "_envs_unique": [0, 1],
                0: [2, 0],
                1: torch.tensor(
                    [
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                    ],
                    dtype=torch.int16,
                    device="cpu",
                ),
            },
            True,
        ),
        (
            "env_grid2d_tetrismini",
            {
                "_active": 1,
                "_envs_unique": [0, 1],
                0: [2, 0],
                1: torch.tensor(
                    [
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                    ],
                    dtype=torch.int16,
                    device="cpu",
                ),
            },
            (1, 1, 0, 0),
            {
                "_active": 1,
                "_envs_unique": [0, 1],
                0: [2, 0],
                1: torch.tensor(
                    [
                        [000, 000, 000, 000],
                        [100, 000, 000, 000],
                        [100, 000, 000, 000],
                        [100, 000, 000, 000],
                        [100, 000, 000, 000],
                    ],
                    dtype=torch.int16,
                    device="cpu",
                ),
            },
            True,
        ),
        (
            "env_grid2d_tetrismini",
            {
                "_active": 1,
                "_envs_unique": [0, 1],
                0: [2, 0],
                1: torch.tensor(
                    [
                        [000, 000, 000, 000],
                        [100, 000, 000, 000],
                        [100, 000, 000, 000],
                        [100, 000, 000, 000],
                        [100, 000, 000, 000],
                    ],
                    dtype=torch.int16,
                    device="cpu",
                ),
            },
            (1, 4, 0, 2),
            {
                "_active": 1,
                "_envs_unique": [0, 1],
                0: [2, 0],
                1: torch.tensor(
                    [
                        [000, 000, 000, 000],
                        [100, 000, 000, 000],
                        [100, 000, 000, 000],
                        [100, 000, 400, 400],
                        [100, 000, 400, 400],
                    ],
                    dtype=torch.int16,
                    device="cpu",
                ),
            },
            True,
        ),
        # EOS from last stage
        (
            "env_grid2d_tetrismini",
            {
                "_active": 1,
                "_envs_unique": [0, 1],
                0: [2, 0],
                1: torch.tensor(
                    [
                        [000, 000, 000, 000],
                        [100, 401, 401, 000],
                        [100, 401, 401, 000],
                        [100, 000, 400, 400],
                        [100, 000, 400, 400],
                    ],
                    dtype=torch.int16,
                    device="cpu",
                ),
            },
            (1, -1, -1, -1),
            {
                "_active": 1,
                "_envs_unique": [0, 1],
                0: [2, 0],
                1: torch.tensor(
                    [
                        [000, 000, 000, 000],
                        [100, 401, 401, 000],
                        [100, 401, 401, 000],
                        [100, 000, 400, 400],
                        [100, 000, 400, 400],
                    ],
                    dtype=torch.int16,
                    device="cpu",
                ),
            },
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
    state_next, action_done, valid = env.step(action)

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
        (
            "env_grid2d_tetrismini",
            {
                "_active": 0,
                "_envs_unique": [0, 1],
                0: [2, 0],
                1: torch.tensor(
                    [
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                    ],
                    dtype=torch.int16,
                    device="cpu",
                ),
            },
            (0, 1, 0, 0),
            {
                "_active": 0,
                "_envs_unique": [0, 1],
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
                    device="cpu",
                ),
            },
            True,
        ),
        (
            "env_grid2d_tetrismini",
            {
                "_active": 1,
                "_envs_unique": [0, 1],
                0: [2, 0],
                1: torch.tensor(
                    [
                        [000, 000, 000, 000],
                        [100, 000, 000, 000],
                        [100, 000, 000, 000],
                        [100, 000, 000, 000],
                        [100, 000, 000, 000],
                    ],
                    dtype=torch.int16,
                    device="cpu",
                ),
            },
            (1, 1, 0, 0),
            {
                "_active": 1,
                "_envs_unique": [0, 1],
                0: [2, 0],
                1: torch.tensor(
                    [
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                    ],
                    dtype=torch.int16,
                    device="cpu",
                ),
            },
            True,
        ),
        (
            "env_grid2d_tetrismini",
            {
                "_active": 1,
                "_envs_unique": [0, 1],
                0: [2, 0],
                1: torch.tensor(
                    [
                        [000, 000, 000, 000],
                        [100, 000, 000, 000],
                        [100, 000, 000, 000],
                        [100, 000, 000, 000],
                        [100, 000, 000, 000],
                    ],
                    dtype=torch.int16,
                    device="cpu",
                ),
            },
            (1, 4, 0, 2),
            {
                "_active": 1,
                "_envs_unique": [0, 1],
                0: [2, 0],
                1: torch.tensor(
                    [
                        [000, 000, 000, 000],
                        [100, 000, 000, 000],
                        [100, 000, 000, 000],
                        [100, 000, 000, 000],
                        [100, 000, 000, 000],
                    ],
                    dtype=torch.int16,
                    device="cpu",
                ),
            },
            False,
        ),
        # Back to source
        (
            "env_grid2d_tetrismini",
            {
                "_active": 0,
                "_envs_unique": [0, 1],
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
                    device="cpu",
                ),
            },
            (0, 1, 0, 0),
            {
                "_active": 0,
                "_envs_unique": [0, 1],
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
                    device="cpu",
                ),
            },
            True,
        ),
        # Action from source must be invalid
        (
            "env_grid2d_tetrismini",
            {
                "_active": 0,
                "_envs_unique": [0, 1],
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
                    device="cpu",
                ),
            },
            (0, 1, 0, 0),
            {
                "_active": 0,
                "_envs_unique": [0, 1],
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
                    device="cpu",
                ),
            },
            False,
        ),
        # Grid EOS from Tetris
        (
            "env_grid2d_tetrismini",
            {
                "_active": 1,
                "_envs_unique": [0, 1],
                0: [2, 0],
                1: torch.tensor(
                    [
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                    ],
                    dtype=torch.int16,
                    device="cpu",
                ),
            },
            (0, 0, 0, 0),
            {
                "_active": 0,
                "_envs_unique": [0, 1],
                0: [2, 0],
                1: torch.tensor(
                    [
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                    ],
                    dtype=torch.int16,
                    device="cpu",
                ),
            },
            True,
        ),
        # Global EOS
        (
            "env_grid2d_tetrismini",
            {
                "_active": 1,
                "_envs_unique": [0, 1],
                0: [2, 0],
                1: torch.tensor(
                    [
                        [000, 000, 000, 000],
                        [100, 401, 401, 000],
                        [100, 401, 401, 000],
                        [100, 000, 400, 400],
                        [100, 000, 400, 400],
                    ],
                    dtype=torch.int16,
                    device="cpu",
                ),
            },
            (1, -1, -1, -1),
            {
                "_active": 1,
                "_envs_unique": [0, 1],
                0: [2, 0],
                1: torch.tensor(
                    [
                        [000, 000, 000, 000],
                        [100, 401, 401, 000],
                        [100, 401, 401, 000],
                        [100, 000, 400, 400],
                        [100, 000, 400, 400],
                    ],
                    dtype=torch.int16,
                    device="cpu",
                ),
            },
            True,
        ),
    ],
)
def test__step_backwards__works_as_expected(
    env, state_from, action, state_next_exp, valid_exp, request
):
    env = request.getfixturevalue(env)
    if action == env.eos and valid_exp:
        done = True
    else:
        done = False
    env.set_state(state_from, done)

    # Check init state
    assert env.equal(env.state, state_from)

    # Perform step
    state_next, action_done, valid = env.step_backwards(action)

    # Check end state
    assert env.equal(env.state, state_next)
    assert env.equal(env.state, state_next_exp)

    # Check action and valid
    assert action_done == action
    assert valid == valid_exp, (state_from, action)


@pytest.mark.repeat(10)
@pytest.mark.parametrize(
    "env", ["env_grid2d_tetrismini", "env_cube_tetris", "env_cube_tetris_grid"]
)
def test__step_random__does_not_crash_from_source(env, request):
    """
    Very low bar test...
    """
    env = request.getfixturevalue(env)
    env.reset()
    env.step_random()
    pass


@pytest.mark.repeat(10)
@pytest.mark.parametrize(
    "env",
    [
        "env_grid2d_tetrismini",
        "env_cube_tetris",
        "env_cube_tetris_grid",
        "env_cube_setgrids",
        "env_cube_setstacks",
    ],
)
def test__trajectory_random__does_not_crash_from_source(env, request):
    """
    Raising the bar...
    """
    env = request.getfixturevalue(env)
    env.reset()
    env.trajectory_random()
    assert True


@pytest.mark.parametrize(
    "env, state, parents_exp, parent_actions_exp",
    [
        # Source
        (
            "env_grid2d_tetrismini",
            {
                "_active": 0,
                "_envs_unique": [0, 1],
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
                    device="cpu",
                ),
            },
            [],
            [],
        ),
        # Intermediate grid
        (
            "env_grid2d_tetrismini",
            {
                "_active": 0,
                "_envs_unique": [0, 1],
                0: [1, 2],
                1: torch.tensor(
                    [
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                    ],
                    dtype=torch.int16,
                    device="cpu",
                ),
            },
            [
                {
                    "_active": 0,
                    "_envs_unique": [0, 1],
                    0: [0, 2],
                    1: torch.tensor(
                        [
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                        ],
                        dtype=torch.int16,
                        device="cpu",
                    ),
                },
                [
                    0,
                    [1, 1],
                    torch.tensor(
                        [
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                        ],
                        dtype=torch.int16,
                        device="cpu",
                    ),
                ],
            ],
            [(0, 1, 0, 0), (0, 0, 1, 0)],
        ),
        # Source Tetris
        (
            "env_grid2d_tetrismini",
            {
                "_active": 1,
                "_envs_unique": [0, 1],
                0: [1, 2],
                1: torch.tensor(
                    [
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                    ],
                    dtype=torch.int16,
                    device="cpu",
                ),
            },
            [
                {
                    "_active": 0,
                    "_envs_unique": [0, 1],
                    0: [1, 2],
                    1: torch.tensor(
                        [
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                            [000, 000, 000, 000],
                        ],
                        dtype=torch.int16,
                        device="cpu",
                    ),
                },
            ],
            [(0, 0, 0, 0)],
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
        # TODO: update if state becomes dictionary
        parents_actions_exp_dict[action] = tuple(parent.copy())
    parents_actions_dict = {}
    for parent, action in zip(parents, parent_actions):
        # TODO: update if state becomes dictionary
        parents_actions_dict[action] = tuple(parent.copy())

    assert all(
        [
            env.equal(a, b)
            for a, b in zip(
                sorted(parents_actions_exp_dict),
                sorted(parents_actions_dict),
            )
        ]
    )


class TestStackGrid2DTetrisMini(common.BaseTestsDiscrete):
    """Common tests for Grid 3x3 -> Tetris-mini."""

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


class TestStackCubeTetris(common.BaseTestsContinuous):
    """Common tests for Cube -> Tetris."""

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


class TestStackCubeTetrisGrid(common.BaseTestsContinuous):
    """Common tests for Cube -> Tetris -> Grid 3x3x3."""

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


class TestStackCubeSetFixGrids(common.BaseTestsContinuous):
    """Common tests for Cube -> SetFix(Grid, Grid)."""

    @pytest.fixture(autouse=True)
    def setup(self, env_cube_setgrids):
        self.env = env_cube_setgrids
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


class TestStackCubeSetFixStacks(common.BaseTestsContinuous):
    """Common tests for Cube -> SetFix(Stack(Cube, Grid), Stack(Grid, Cube))."""

    @pytest.fixture(autouse=True)
    def setup(self, env_cube_setstacks):
        self.env = env_cube_setstacks
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
