import warnings

import common
import numpy as np
import pytest
import torch
from torch import Tensor

from gflownet.envs.cube import ContinuousCube
from gflownet.envs.grid import Grid
from gflownet.envs.set import SetFix, SetFlex, make_set
from gflownet.envs.stack import Stack
from gflownet.envs.tetris import Tetris
from gflownet.utils.common import copy, tbool, tfloat


@pytest.fixture
def env_flex_two_grids():
    return make_set(
        is_flexible=True,
        envs_unique=(Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),),
        max_elements=2,
        do_random_subenvs=True,
    )


@pytest.fixture
def env_flex_missing_max_elements_two_grids():
    return make_set(
        is_flexible=True,
        envs_unique=(Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),),
        do_random_subenvs=True,
    )


@pytest.fixture
def env_flex_three_cubes():
    return make_set(
        is_flexible=True,
        envs_unique=(ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),),
        max_elements=3,
        do_random_subenvs=True,
    )


@pytest.fixture
def env_flex_three_cubes_grids():
    return make_set(
        is_flexible=True,
        envs_unique=(
            ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
            Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),
        ),
        max_elements=3,
        do_random_subenvs=True,
    )


@pytest.fixture
def env_flex_missing_max_elements_three_cubes_grids():
    return make_set(
        is_flexible=True,
        envs_unique=(
            ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
            Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),
        ),
        do_random_subenvs=True,
    )


# Initialize the environment without passing envs_unique
@pytest.fixture
def env_flex_five_cubes_grids_implicit_unique():
    return make_set(
        is_flexible=True,
        subenvs=(
            ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
            Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),
            Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),
            ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
        ),
        max_elements=5,
    )


@pytest.fixture
def env_flex_missing_max_elements_five_cubes_grids_implicit_unique():
    return make_set(
        is_flexible=True,
        subenvs=(
            ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
            Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),
            Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),
            ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
        ),
    )


@pytest.fixture
def env_flex_two_stacks():
    return make_set(
        is_flexible=True,
        envs_unique=(
            Stack(
                subenvs=(
                    ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
                    Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),
                )
            ),
        ),
        max_elements=2,
        do_random_subenvs=True,
    )


@pytest.fixture
def env_flex_three_stacks_diff():
    return make_set(
        is_flexible=True,
        envs_unique=(
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
        ),
        max_elements=3,
        do_random_subenvs=True,
    )


@pytest.fixture
def env_fix_grid2d_tetrismini():
    return make_set(
        is_flexible=False,
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
        ),
    )


@pytest.fixture
def env_fix_cube_tetris():
    return make_set(
        is_flexible=False,
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
        ),
    )


@pytest.fixture
def env_fix_cube_tetris_grid():
    return make_set(
        is_flexible=False,
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
        ),
    )


@pytest.fixture
def env_fix_tetris_grid_cube():
    return make_set(
        is_flexible=False,
        subenvs=(
            Tetris(
                width=4,
                height=5,
                pieces=["I", "O"],
                rotations=[0],
                allow_eos_before_full=True,
                device="cpu",
            ),
            Grid(n_dim=3, length=3, cell_min=-1.0, cell_max=1.0),
            ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
        ),
    )


@pytest.fixture
def env_fix_two_grids():
    return make_set(
        is_flexible=False,
        subenvs=(
            Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),
            Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),
        ),
    )


@pytest.fixture
def env_fix_three_cubes():
    return make_set(
        is_flexible=False,
        subenvs=(
            ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
            ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
            ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
        ),
    )


@pytest.fixture
def env_fix_two_grids_three_cubes():
    return make_set(
        is_flexible=False,
        subenvs=(
            Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),
            Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),
            ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
            ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
            ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
        ),
    )


@pytest.fixture
def env_fix_cube2d_cube3d():
    return make_set(
        is_flexible=False,
        subenvs=(
            ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
            ContinuousCube(n_dim=3, n_comp=3, min_incr=0.1),
        ),
    )


@pytest.fixture
def env_fix_two_cubes2d_one_cube3d():
    return make_set(
        is_flexible=False,
        subenvs=(
            ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
            ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
            ContinuousCube(n_dim=3, n_comp=3, min_incr=0.1),
        ),
    )


@pytest.fixture
def env_fix_stacks_equal():
    return make_set(
        is_flexible=False,
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
        ),
    )


@pytest.fixture
def env_fix_missing_subenvs():
    return make_set(
        is_flexible=False,
    )


@pytest.fixture
def env_fix_stacks_diff():
    return make_set(
        is_flexible=False,
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
        ),
    )


@pytest.mark.parametrize(
    "env",
    [
        "env_flex_two_grids",
        "env_flex_three_cubes",
        "env_flex_three_cubes_grids",
        "env_flex_five_cubes_grids_implicit_unique",
        "env_flex_two_stacks",
        "env_flex_three_stacks_diff",
    ],
)
def test__environment__flex_initializes_properly(env, request):
    env = request.getfixturevalue(env)
    assert env.__class__ == SetFlex


@pytest.mark.parametrize(
    "env",
    [
        "env_fix_grid2d_tetrismini",
        "env_fix_cube_tetris",
        "env_fix_cube_tetris_grid",
        "env_fix_tetris_grid_cube",
        "env_fix_two_grids",
        "env_fix_three_cubes",
        "env_fix_two_grids_three_cubes",
        "env_fix_cube2d_cube3d",
        "env_fix_two_cubes2d_one_cube3d",
        "env_fix_stacks_equal",
        "env_fix_stacks_diff",
    ],
)
def test__environment__fix_initializes_properly(env, request):
    env = request.getfixturevalue(env)
    assert env.__class__ == SetFix


@pytest.mark.parametrize(
    "env",
    [
        "env_flex_missing_max_elements_two_grids",
        "env_flex_missing_max_elements_three_cubes_grids",
        "env_flex_missing_max_elements_five_cubes_grids_implicit_unique",
    ],
)
def test__environment__flex_catches_missing_max_elements(env, request):
    with pytest.raises(
        Exception, match="max_elements must be defined to use the SetFlex"
    ):
        env = request.getfixturevalue(env)


@pytest.mark.parametrize(
    "env",
    [
        "env_fix_missing_subenvs",
    ],
)
def test__environment__fix_catches_missing_max_elements(env, request):
    with pytest.raises(Exception, match="subenvs must be defined to use the SetFix"):
        env = request.getfixturevalue(env)


@pytest.mark.parametrize(
    "env, state_x, state_y, equal_exp",
    [
        (
            "env_fix_two_grids",
            {
                "_active": -1,
                "_toggle": 0,
                "_dones": [0, 0],
                "_envs_unique": [0, 0],
                "_keys": [0, 1],
                0: [0, 0],
                1: [0, 0],
            },
            {
                "_active": -1,
                "_toggle": 0,
                "_dones": [0, 0],
                "_envs_unique": [0, 0],
                "_keys": [0, 1],
                0: [0, 0],
                1: [0, 0],
            },
            True,
        ),
        (
            "env_fix_two_grids",
            {
                "_active": 1,
                "_toggle": 0,
                "_dones": [1, 0],
                "_envs_unique": [0, 0],
                "_keys": [0, 1],
                0: [1, 2],
                1: [2, 1],
            },
            {
                "_active": 1,
                "_toggle": 0,
                "_dones": [1, 0],
                "_envs_unique": [0, 0],
                "_keys": [0, 1],
                0: [1, 2],
                1: [2, 1],
            },
            True,
        ),
        # Permute substates and keys
        (
            "env_fix_two_grids",
            {
                "_active": 1,
                "_toggle": 0,
                "_dones": [1, 0],
                "_envs_unique": [0, 0],
                "_keys": [1, 0],
                0: [2, 1],
                1: [1, 2],
            },
            {
                "_active": 1,
                "_toggle": 0,
                "_dones": [1, 0],
                "_envs_unique": [0, 0],
                "_keys": [0, 1],
                0: [1, 2],
                1: [2, 1],
            },
            True,
        ),
        # Permute keys without permuting substates
        (
            "env_fix_two_grids",
            {
                "_active": 1,
                "_toggle": 0,
                "_dones": [1, 0],
                "_envs_unique": [0, 0],
                "_keys": [1, 0],
                0: [2, 1],
                1: [1, 2],
            },
            {
                "_active": 1,
                "_toggle": 0,
                "_dones": [1, 0],
                "_envs_unique": [0, 0],
                "_keys": [0, 1],
                0: [2, 1],
                1: [1, 2],
            },
            False,
        ),
        (
            "env_fix_two_grids_three_cubes",
            {
                "_active": 3,
                "_toggle": 0,
                "_dones": [1, 1, 1, 0, 0],
                "_envs_unique": [0, 0, 1, 1, 1],
                "_keys": [0, 1, 2, 3, 4],
                0: [1, 2],
                1: [2, 1],
                2: [0.44, 0.55],
                3: [0.33, 0.22],
                4: [-1, -1],
            },
            {
                "_active": 3,
                "_toggle": 0,
                "_dones": [1, 1, 1, 0, 0],
                "_envs_unique": [0, 0, 1, 1, 1],
                "_keys": [0, 1, 2, 3, 4],
                0: [1, 2],
                1: [2, 1],
                2: [0.44, 0.55],
                3: [0.33, 0.22],
                4: [-1, -1],
            },
            True,
        ),
    ],
)
def test__equal__of_set_behaves_as_expected(env, state_x, state_y, equal_exp, request):
    env = request.getfixturevalue(env)
    assert env.equal(state_x, state_y) == equal_exp


@pytest.mark.repeat(10)
@pytest.mark.parametrize(
    "env, state, idx_unique, done_only, permutations_keys",
    [
        (
            "env_fix_two_grids",
            {
                "_active": -1,
                "_toggle": 0,
                "_dones": [0, 0],
                "_envs_unique": [0, 0],
                "_keys": [0, 1],
                0: [0, 0],
                1: [0, 0],
            },
            0,
            False,
            [[0, 1], [1, 0]],
        ),
        (
            "env_fix_two_grids",
            {
                "_active": -1,
                "_toggle": 0,
                "_dones": [1, 1],
                "_envs_unique": [0, 0],
                "_keys": [0, 1],
                0: [1, 2],
                1: [2, 1],
            },
            0,
            True,
            [[0, 1], [1, 0]],
        ),
    ],
)
def test__permute_substates__behaves_as_expected(
    env, state, idx_unique, done_only, permutations_keys, request
):
    env = request.getfixturevalue(env)
    state_orig = copy(state)
    state_permuted, idx_last_done = env._permute_substates(idx_unique, state, done_only)
    assert state_permuted["_keys"] in permutations_keys
    assert env.equal(state_orig, state_permuted)
