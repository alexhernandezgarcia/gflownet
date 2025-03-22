"""
Note that the common test test__gflownet_minimal_runs is disabled because the SetFlex
is not designed to be used in isolation but as part of a macro-environment that sets
its sub-environments for each trajectory. Otherwise, training is bound to crash because
the transtions from the (generic) source state to the first state would be incorrect,
even if random sub-environments are sampled at initialization.
"""

import warnings

import common
import numpy as np
import pytest
import torch
from torch import Tensor

from gflownet.envs.constant import Constant
from gflownet.envs.cube import ContinuousCube
from gflownet.envs.dummy import Dummy
from gflownet.envs.grid import Grid
from gflownet.envs.set import SetFlex
from gflownet.envs.stack import Stack
from gflownet.utils.common import copy, tbool, tfloat


@pytest.fixture
def env_two_grids():
    return SetFlex(
        envs_unique=(Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),),
        max_elements=2,
        do_random_subenvs=True,
    )


@pytest.fixture
def env_three_cubes():
    return SetFlex(
        envs_unique=(ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),),
        max_elements=3,
        do_random_subenvs=True,
    )


@pytest.fixture
def env_three_cubes_grids():
    return SetFlex(
        envs_unique=(
            ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
            Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),
        ),
        max_elements=3,
        do_random_subenvs=True,
    )


# Initialize the environment without passing envs_unique
@pytest.fixture
def env_five_cubes_grids_implicit_unique():
    return SetFlex(
        subenvs=(
            ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
            Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),
            Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),
            ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
        ),
        max_elements=5,
    )


@pytest.fixture
def env_two_stacks():
    return SetFlex(
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
def env_three_stacks_diff():
    return SetFlex(
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
def env_three_orbit_like_stacks():
    return SetFlex(
        envs_unique=(
            Stack(
                subenvs=(
                    Dummy(state=[1], is_flexible=True),
                    ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
                )
            ),
            Stack(
                subenvs=(
                    Dummy(state=[2], is_flexible=True),
                    Constant(state=[-1, -1]),
                )
            ),
        ),
        subenvs=(
            Stack(
                subenvs=(
                    Dummy(state=[3], is_flexible=True),
                    ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
                )
            ),
            Stack(
                subenvs=(
                    Dummy(state=[4], is_flexible=True),
                    Constant(state=[-1, -1]),
                )
            ),
            Stack(
                subenvs=(
                    Dummy(state=[5], is_flexible=True),
                    ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
                )
            ),
        ),
        max_elements=3,
    )


@pytest.mark.parametrize(
    "env",
    [
        "env_two_grids",
        "env_three_cubes",
        "env_three_cubes_grids",
        "env_five_cubes_grids_implicit_unique",
        "env_two_stacks",
        "env_three_stacks_diff",
    ],
)
def test__environment__initializes_properly(env, request):
    env = request.getfixturevalue(env)
    assert True


@pytest.mark.parametrize(
    "env, n_unique_envs",
    [
        ("env_two_grids", 1),
        ("env_three_cubes", 1),
        ("env_three_cubes_grids", 2),
        ("env_five_cubes_grids_implicit_unique", 2),
        ("env_two_stacks", 1),
        ("env_three_stacks_diff", 2),
    ],
)
def test__number_of_unique_envs_is_correct(env, request, n_unique_envs):
    env = request.getfixturevalue(env)
    assert env.n_unique_envs == n_unique_envs


@pytest.mark.parametrize(
    "env, is_continuous",
    [
        ("env_two_grids", False),
        ("env_three_cubes", True),
        ("env_three_cubes_grids", True),
        ("env_five_cubes_grids_implicit_unique", True),
        ("env_two_stacks", True),
        ("env_three_stacks_diff", True),
    ],
)
def test__environment__is_continuous(env, is_continuous, request):
    env = request.getfixturevalue(env)
    assert env.continuous == is_continuous


@pytest.mark.parametrize(
    "env, mask_dim_expected",
    [
        ("env_two_grids", max([3, 2 + 1]) + 2),
        ("env_three_cubes", max([5, 3 + 1]) + 3),
        ("env_three_cubes_grids", max([5, 3, 3 + 1]) + 3),
        ("env_five_cubes_grids_implicit_unique", max([5, 3, 5 + 1]) + 5),
        ("env_two_stacks", max([max([5, 3]) + 2, max([5, 3]) + 2, 2 + 1]) + 2),
        ("env_three_stacks_diff", max([max([5, 3]) + 2, max([5, 3]) + 2, 3 + 1]) + 3),
    ],
)
def test__mask_dim__is_as_expected(env, request, mask_dim_expected):
    env = request.getfixturevalue(env)
    assert env.mask_dim == mask_dim_expected


@pytest.mark.parametrize(
    "env, action_space",
    [
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
        (
            "env_three_cubes",
            [
                # fmt: off
                # Activate subenvs
                (-1, 0, 0, 0),
                (-1, 1, 0, 0),
                (-1, 2, 0, 0),
                # EOS
                (-1, -1, -1, -1),
                # Cube
                (0, 0.0, 0.0, 0),
                (0, 0.0, 0.0, 1),
                (0, np.inf, np.inf, np.inf),
                # fmt: on
            ],
        ),
        (
            "env_three_cubes_grids",
            [
                # fmt: off
                # Activate subenvs
                (-1, 0, 0, 0),
                (-1, 1, 0, 0),
                (-1, 2, 0, 0),
                # EOS
                (-1, -1, -1, -1),
                # Cube
                (0, 0.0, 0.0, 0),
                (0, 0.0, 0.0, 1),
                (0, np.inf, np.inf, np.inf),
                # Grid
                (1, 0, 0, 0),
                (1, 1, 0, 0),
                (1, 0, 1, 0),
                # fmt: on
            ],
        ),
        (
            "env_five_cubes_grids_implicit_unique",
            [
                # fmt: off
                # Activate subenvs
                (-1, 0, 0, 0),
                (-1, 1, 0, 0),
                (-1, 2, 0, 0),
                (-1, 3, 0, 0),
                (-1, 4, 0, 0),
                # EOS
                (-1, -1, -1, -1),
                # Cube
                (0, 0.0, 0.0, 0),
                (0, 0.0, 0.0, 1),
                (0, np.inf, np.inf, np.inf),
                # Grid
                (1, 0, 0, 0),
                (1, 1, 0, 0),
                (1, 0, 1, 0),
                # fmt: on
            ],
        ),
    ],
)
def test__get_action_space__returns_expected(env, action_space, request):
    env = request.getfixturevalue(env)
    assert set(action_space) == set(env.action_space)


@pytest.mark.parametrize(
    "env, subenvs, indices_unique",
    [
        ("env_two_grids", (Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),), [0]),
        (
            "env_two_grids",
            (
                Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),
                Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),
            ),
            [0, 0],
        ),
        ("env_three_cubes", (ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),), [0]),
        (
            "env_three_cubes",
            (
                ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
                ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
            ),
            [0, 0],
        ),
        (
            "env_three_cubes",
            (
                ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
                ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
                ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
            ),
            [0, 0, 0],
        ),
        (
            "env_three_cubes_grids",
            (Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),),
            [1],
        ),
        (
            "env_three_cubes_grids",
            (ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),),
            [0],
        ),
        (
            "env_three_cubes_grids",
            (
                Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),
                ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
            ),
            [1, 0],
        ),
        (
            "env_three_cubes_grids",
            (
                ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
                Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),
                ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
            ),
            [0, 1, 0],
        ),
        (
            "env_two_stacks",
            (
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
            [0, 0],
        ),
        (
            "env_three_stacks_diff",
            (
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
                Stack(
                    subenvs=(
                        Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),
                        ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
                    )
                ),
            ),
            [0, 0, 1],
        ),
    ],
)
def test__compute_unique_indices_of_subenvs__returns_expected(
    env, subenvs, indices_unique, request
):
    env = request.getfixturevalue(env)
    assert env._compute_unique_indices_of_subenvs(subenvs) == indices_unique


@pytest.mark.parametrize(
    "env, subenvs, state",
    [
        (
            "env_two_grids",
            (Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),),
            [[-1, 0, [0, 1], [0, -1]], {0: [0, 0]}],
        ),
        (
            "env_two_grids",
            (
                Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),
                Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),
            ),
            [[-1, 0, [0, 0], [0, 0]], {0: [0, 0], 1: [0, 0]}],
        ),
        (
            "env_three_cubes",
            (ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),),
            [[-1, 0, [0, 1, 1], [0, -1, -1]], {0: [-1, -1]}],
        ),
        (
            "env_three_cubes",
            (
                ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
                ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
            ),
            [[-1, 0, [0, 0, 1], [0, 0, -1]], {0: [-1, -1], 1: [-1, -1]}],
        ),
        (
            "env_three_cubes",
            (
                ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
                ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
                ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
            ),
            [[-1, 0, [0, 0, 0], [0, 0, 0]], {0: [-1, -1], 1: [-1, -1], 2: [-1, -1]}],
        ),
        (
            "env_three_cubes_grids",
            (Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),),
            [[-1, 0, [0, 1, 1], [1, -1, -1]], {0: [0, 0]}],
        ),
        (
            "env_three_cubes_grids",
            (ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),),
            [[-1, 0, [0, 1, 1], [0, -1, -1]], {0: [-1, -1]}],
        ),
        (
            "env_three_cubes_grids",
            (
                Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),
                ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
            ),
            [[-1, 0, [0, 0, 1], [1, 0, -1]], {0: [0, 0], 1: [-1, -1]}],
        ),
        (
            "env_three_cubes_grids",
            (
                ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
                Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),
                ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
            ),
            [[-1, 0, [0, 0, 0], [0, 1, 0]], {0: [-1, -1], 1: [0, 0], 2: [-1, -1]}],
        ),
        (
            "env_two_stacks",
            (
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
            [
                [-1, 0, [0, 0], [0, 0]],
                {
                    0: [0, [-1, -1], [0, 0]],
                    1: [0, [-1, -1], [0, 0]],
                },
            ],
        ),
        (
            "env_three_stacks_diff",
            (
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
                Stack(
                    subenvs=(
                        Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),
                        ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
                    )
                ),
            ),
            [
                [-1, 0, [0, 0, 0], [0, 0, 1]],
                {
                    0: [0, [-1, -1], [0, 0]],
                    1: [0, [-1, -1], [0, 0]],
                    2: [0, [0, 0], [-1, -1]],
                },
            ],
        ),
    ],
)
def test__set_subenvs__applies_changes_as_expected(env, subenvs, state, request):
    env = request.getfixturevalue(env)
    env.set_subenvs(subenvs)
    # Check self.subenvs
    assert all([s_env == s_arg for s_env, s_arg in zip(env.subenvs, subenvs)])
    # Check state
    assert env.equal(env.state, state)


@pytest.mark.parametrize(
    "env, subenvs, state, is_source",
    [
        (
            "env_two_grids",
            (Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),),
            [[-1, 0, [0, 1], [0, -1]], {0: [0, 0]}],
            True,
        ),
        (
            "env_two_grids",
            (Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),),
            [[0, 1, [0, 1], [0, -1]], {0: [0, 0]}],
            False,
        ),
        (
            "env_two_grids",
            (Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),),
            [[-1, 0, [1, 1], [0, -1]], {0: [0, 0]}],
            False,
        ),
        (
            "env_two_grids",
            (
                Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),
                Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),
            ),
            [[-1, 0, [0, 0], [0, 0]], {0: [0, 0], 1: [0, 0]}],
            True,
        ),
        (
            "env_two_grids",
            (
                Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),
                Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),
            ),
            [[-1, 0, [0, 1], [0, 0]], {0: [0, 0], 1: [0, 0]}],
            False,
        ),
        (
            "env_two_grids",
            (
                Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),
                Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),
            ),
            [[-1, 0, [0, 1], [0, 0]], {0: [0, 0]}],
            False,
        ),
        (
            "env_three_cubes",
            (ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),),
            [[-1, 0, [0, 1, 1], [0, -1, -1]], {0: [-1, -1]}],
            True,
        ),
        (
            "env_three_cubes",
            (ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),),
            [[0, 1, [0, 1, 1], [0, -1, -1]], {0: [-1, -1]}],
            False,
        ),
        (
            "env_three_cubes",
            (
                ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
                ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
            ),
            [[-1, 0, [0, 0, 1], [0, 0, -1]], {0: [-1, -1], 1: [-1, -1]}],
            True,
        ),
        (
            "env_three_cubes",
            (
                ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
                ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
            ),
            [[-1, 0, [0, 0, 1], [0, 0, -1]], {0: [0.1, 0.2], 1: [-1, -1]}],
            False,
        ),
        (
            "env_three_cubes",
            (
                ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
                ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
                ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
            ),
            [[-1, 0, [0, 0, 0], [0, 0, 0]], {0: [-1, -1], 1: [-1, -1], 2: [-1, -1]}],
            True,
        ),
        (
            "env_three_cubes_grids",
            (Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),),
            [[-1, 0, [0, 1, 1], [1, -1, -1]], {0: [0, 0]}],
            True,
        ),
        (
            "env_three_cubes_grids",
            (Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),),
            [[-1, 0, [1, 1, 1], [1, -1, -1]], {0: [0, 0]}],
            False,
        ),
        (
            "env_three_cubes_grids",
            (ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),),
            [[-1, 0, [0, 1, 1], [0, -1, -1]], {0: [-1, -1]}],
            True,
        ),
        (
            "env_three_cubes_grids",
            (ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),),
            [[-1, 0, [0, 1, 1], [1, -1, -1]], {0: [-1, -1]}],
            False,
        ),
        (
            "env_three_cubes_grids",
            (
                Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),
                ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
            ),
            [[-1, 0, [0, 0, 1], [1, 0, -1]], {0: [0, 0], 1: [-1, -1]}],
            True,
        ),
        (
            "env_three_cubes_grids",
            (
                ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
                Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),
                ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
            ),
            [[-1, 0, [0, 0, 0], [0, 1, 0]], {0: [-1, -1], 1: [0, 0], 2: [-1, -1]}],
            True,
        ),
        (
            "env_three_cubes_grids",
            (
                ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
                Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),
                ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
            ),
            [[-1, 0, [0, 1, 0], [0, 1, 0]], {0: [-1, -1], 1: [0, 0], 2: [-1, -1]}],
            False,
        ),
        (
            "env_two_stacks",
            (
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
            [
                [-1, 0, [0, 0], [0, 0]],
                {
                    0: [0, [-1, -1], [0, 0]],
                    1: [0, [-1, -1], [0, 0]],
                },
            ],
            True,
        ),
        (
            "env_two_stacks",
            (
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
            [
                [-1, 0, [0, 0], [0, 0]],
                {
                    0: [0, [-1, -1], [0, 0]],
                },
            ],
            False,
        ),
        (
            "env_three_stacks_diff",
            (
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
                Stack(
                    subenvs=(
                        Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),
                        ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
                    )
                ),
            ),
            [
                [-1, 0, [0, 0, 0], [0, 0, 1]],
                {
                    0: [0, [-1, -1], [0, 0]],
                    1: [0, [-1, -1], [0, 0]],
                    2: [0, [0, 0], [-1, -1]],
                },
            ],
            True,
        ),
        (
            "env_three_stacks_diff",
            (
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
                Stack(
                    subenvs=(
                        Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),
                        ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
                    )
                ),
            ),
            [
                [-1, 0, [0, 0, 0], [0, 0, 1]],
                {
                    0: [0, [-1, -1], [0, 0]],
                    1: [0, [-1, -1], [0, 0]],
                    2: [1, [0, 0], [-1, -1]],
                },
            ],
            False,
        ),
        (
            "env_three_orbit_like_stacks",
            (
                Stack(
                    subenvs=(
                        Dummy(state=[3], is_flexible=True),
                        ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
                    )
                ),
                Stack(
                    subenvs=(
                        Dummy(state=[4], is_flexible=True),
                        Constant(state=[-1, -1]),
                    )
                ),
                Stack(
                    subenvs=(
                        Dummy(state=[5], is_flexible=True),
                        ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
                    )
                ),
            ),
            [
                [-1, 0, [0, 0, 0], [0, 1, 0]],
                {
                    0: [0, [3], [-1, -1]],
                    1: [0, [4], [-1, -1]],
                    2: [0, [5], [-1, -1]],
                },
            ],
            True,
        ),
        (
            "env_three_orbit_like_stacks",
            (
                Stack(
                    subenvs=(
                        Dummy(state=[3], is_flexible=True),
                        ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
                    )
                ),
                Stack(
                    subenvs=(
                        Dummy(state=[4], is_flexible=True),
                        Constant(state=[-1, -1]),
                    )
                ),
                Stack(
                    subenvs=(
                        Dummy(state=[5], is_flexible=True),
                        ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
                    )
                ),
            ),
            [
                [-1, 0, [0, 0, 0], [0, 1, 0]],
                {
                    0: [0, [5], [-1, -1]],
                    1: [0, [6], [-1, -1]],
                    2: [0, [7], [-1, -1]],
                },
            ],
            True,
        ),
        (
            "env_three_orbit_like_stacks",
            (
                Stack(
                    subenvs=(
                        Dummy(state=[3], is_flexible=True),
                        ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
                    )
                ),
            ),
            [
                [-1, 0, [0, 1, 1], [0, -1, -1]],
                {
                    0: [0, [5], [-1, -1]],
                },
            ],
            True,
        ),
        (
            "env_three_orbit_like_stacks",
            (
                Stack(
                    subenvs=(
                        Dummy(state=[3], is_flexible=True),
                        ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
                    )
                ),
            ),
            [
                [0, 1, [0, 1, 1], [0, -1, -1]],
                {
                    0: [0, [5], [-1, -1]],
                },
            ],
            False,
        ),
    ],
)
def test__is_source__returns_expected(env, subenvs, state, is_source, request):
    env = request.getfixturevalue(env)
    env.set_subenvs(subenvs)
    assert env.is_source(state) == is_source
    # Some states are invalid given the sub-environments
    try:
        env.set_state(state)
        assert env.is_source() == is_source
    except:
        assert True


@pytest.mark.parametrize(
    "env, state, done, subenvs",
    [
        (
            "env_two_grids",
            # Source
            [[-1, 0, [0, 1], [0, -1]], {0: [0, 0]}],
            False,
            (Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),),
        ),
        (
            "env_two_grids",
            # Intermediate
            [[-1, 0, [0, 1], [0, -1]], {0: [1, 2]}],
            False,
            (Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),),
        ),
        (
            "env_two_grids",
            # Done
            [[-1, 0, [1, 1], [0, -1]], {0: [1, 2]}],
            True,
            (Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),),
        ),
        (
            "env_two_grids",
            # Source
            [[-1, 0, [0, 0], [0, 0]], {0: [0, 0], 1: [0, 0]}],
            False,
            (
                Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),
                Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),
            ),
        ),
        (
            "env_two_grids",
            # Intermediate
            [[0, 1, [0, 1], [0, 0]], {0: [0, 2], 1: [1, 1]}],
            False,
            (
                Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),
                Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),
            ),
        ),
        (
            "env_three_cubes_grids",
            # Source
            [[-1, 0, [0, 1, 1], [0, -1, -1]], {0: [-1, -1]}],
            False,
            (ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),),
        ),
        (
            "env_three_cubes_grids",
            # Done
            [[-1, 0, [1, 1, 1], [0, -1, -1]], {0: [0.1234, 0.4321]}],
            False,
            (ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),),
        ),
        (
            "env_three_cubes_grids",
            # Source
            [[-1, 0, [0, 0, 1], [1, 0, -1]], {0: [0, 0], 1: [-1, -1]}],
            False,
            (
                Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),
                ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
            ),
        ),
        (
            "env_three_cubes_grids",
            # Intermediate
            [[0, 1, [0, 1, 1], [1, 0, -1]], {0: [1, 1], 1: [0.1224, 0.4321]}],
            False,
            (
                Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),
                ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
            ),
        ),
    ],
)
def test__set_state__sets_state_dones_and_subenvs(env, state, done, subenvs, request):
    env = request.getfixturevalue(env)
    env.set_state(state, done)

    # Check global state
    assert env.equal(env.state, state)

    # Check subenvs
    assert all(
        [
            (type(s_env), tuple(s_env.action_space))
            == (type(s_arg), tuple(s_arg.action_space))
            for s_env, s_arg in zip(env.subenvs, subenvs)
        ]
    )

    # Check states of subenvs
    for idx, subenv in enumerate(env.subenvs):
        assert env.equal(subenv.state, env._get_substate(state, idx))

    # Check global done
    assert env.done == done

    # Check dones
    dones = env._get_dones()
    for subenv, done in zip(env.subenvs, dones):
        assert subenv.done == done


@pytest.mark.parametrize(
    "env, subenvs, state, mask_exp",
    [
        # Source, one env
        # The main mask is the Set mask: can only activate subenv 0 because there is
        # only one. EOS invalid
        (
            "env_two_grids",
            (Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),),
            [
                [-1, 0, [0, 1], [0, -1]],
                {
                    0: [0, 0],
                },
            ],
            # fmt: off
            [
                False, False, # ACTIVE SUBENV
                False, True, True, # MASK
            ]
            # fmt: on
        ),
        # Source, two envs
        # The main mask is the Set mask: can activate any of the two envs. EOS invalid.
        (
            "env_two_grids",
            (
                Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),
                Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),
            ),
            [
                [-1, 0, [0, 0], [0, 0]],
                {
                    0: [0, 0],
                    1: [0, 0],
                },
            ],
            # fmt: off
            [
                False, False, # ACTIVE SUBENV
                False, False, True, # MASK
            ]
            # fmt: on
        ),
        # Intermediate, two envs
        (
            "env_two_grids",
            (
                Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),
                Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),
            ),
            [
                [0, 1, [0, 0], [0, 0]],
                {
                    0: [1, 2],
                    1: [0, 1],
                },
            ],
            # fmt: off
            [
                True, False, # ACTIVE SUBENV
                True, False, False, # MASK
            ]
            # fmt: on
        ),
        # Intermediate, two envs
        (
            "env_two_grids",
            (
                Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),
                Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),
            ),
            [
                [1, 1, [0, 0], [0, 0]],
                {
                    0: [1, 2],
                    1: [0, 1],
                },
            ],
            # fmt: off
            [
                False, True, # ACTIVE SUBENV
                False, False, False, # MASK
            ]
            # fmt: on
        ),
        # Source
        # There are two envs of three, so activating last one is invalid. EOS invalid.
        (
            "env_three_cubes_grids",
            (
                Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),
                ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
            ),
            [
                [-1, 0, [0, 0, 1], [1, 0, -1]],
                {
                    0: [0, 0],
                    1: [-1, -1],
                },
            ],
            # fmt: off
            [
                False, False, False, # ACTIVE SUBENV
                False, False, True, True, # MASK
                False, # PAD
            ]
            # fmt: on
        ),
        # Intermediate, before activating subenv
        (
            "env_three_cubes_grids",
            (
                Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),
                ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
            ),
            [
                [-1, 0, [0, 0, 1], [1, 0, -1]],
                {
                    0: [1, 1],
                    1: [0.25, 0.15],
                },
            ],
            # fmt: off
            [
                False, False, False, # ACTIVE SUBENV
                False, False, True, True, # MASK
                False, # PAD
            ]
            # fmt: on
        ),
        # Intermediate, before subenv action
        (
            "env_three_cubes_grids",
            (
                Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),
                ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
            ),
            [
                [0, 1, [0, 0, 1], [1, 0, -1]],
                {
                    0: [1, 1],
                    1: [0.25, 0.15],
                },
            ],
            # fmt: off
            [
                True, False, False, # ACTIVE SUBENV
                False, False, False, # MASK
                False, False, # PAD
            ]
            # fmt: on
        ),
        # Intermediate, after subenv action
        (
            "env_three_cubes_grids",
            (
                Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),
                ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
            ),
            [
                [0, 0, [0, 0, 1], [1, 0, -1]],
                {
                    0: [2, 1],
                    1: [0.25, 0.15],
                },
            ],
            # fmt: off
            [
                False, False, False, # ACTIVE SUBENV
                False, True, True, True, # MASK
                False, # PAD
            ]
            # fmt: on
        ),
        # Intermediate, one subenv is done
        (
            "env_three_cubes_grids",
            (
                Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),
                ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
            ),
            [
                [-1, 0, [0, 1, 1], [1, 0, -1]],
                {
                    0: [2, 1],
                    1: [0.25, 0.15],
                },
            ],
            # fmt: off
            [
                False, False, False, # ACTIVE SUBENV
                False, True, True, True, # MASK
                False, # PAD
            ]
            # fmt: on
        ),
        # All subenvs are done
        (
            "env_three_cubes_grids",
            (
                Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),
                ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
            ),
            [
                [-1, 0, [1, 1, 1], [1, 0, -1]],
                {
                    0: [2, 1],
                    1: [0.25, 0.15],
                },
            ],
            # fmt: off
            [
                False, False, False, # ACTIVE SUBENV
                True, True, True, False, # MASK
                False, # PAD
            ]
            # fmt: on
        ),
    ],
)
def test__get_mask_invalid_actions_forward__returns_expected(
    env, subenvs, state, mask_exp, request
):
    env = request.getfixturevalue(env)
    env.set_subenvs(subenvs)
    # Passing state
    mask = env.get_mask_invalid_actions_forward(state, done=False)
    assert mask == mask_exp
    # State from env
    env.set_state(state, done=False)
    mask = env.get_mask_invalid_actions_forward()
    assert mask == mask_exp


@pytest.mark.parametrize(
    "env, state, mask_exp",
    [
        # Source, one env
        # The main mask is the Set mask: can only activate subenv 0 because there is
        # only one. EOS invalid
        (
            "env_two_grids",
            [
                [-1, 0, [0, 1], [0, -1]],
                {
                    0: [0, 0],
                },
            ],
            # fmt: off
            [
                False, False, # ACTIVE SUBENV
                False, True, True, # MASK
            ]
            # fmt: on
        ),
        # Source, two envs
        # The main mask is the Set mask: can activate any of the two envs. EOS invalid.
        (
            "env_two_grids",
            [
                [-1, 0, [0, 0], [0, 0]],
                {
                    0: [0, 0],
                    1: [0, 0],
                },
            ],
            # fmt: off
            [
                False, False, # ACTIVE SUBENV
                False, False, True, # MASK
            ]
            # fmt: on
        ),
        # Intermediate, two envs
        (
            "env_two_grids",
            [
                [0, 1, [0, 0], [0, 0]],
                {
                    0: [1, 2],
                    1: [0, 1],
                },
            ],
            # fmt: off
            [
                True, False, # ACTIVE SUBENV
                True, False, False, # MASK
            ]
            # fmt: on
        ),
        # Intermediate, two envs
        (
            "env_two_grids",
            [
                [1, 1, [0, 0], [0, 0]],
                {
                    0: [1, 2],
                    1: [0, 1],
                },
            ],
            # fmt: off
            [
                False, True, # ACTIVE SUBENV
                False, False, False, # MASK
            ]
            # fmt: on
        ),
        # Source
        # There are two envs of three, so activating last one is invalid. EOS invalid.
        (
            "env_three_cubes_grids",
            [
                [-1, 0, [0, 0, 1], [1, 0, -1]],
                {
                    0: [0, 0],
                    1: [-1, -1],
                },
            ],
            # fmt: off
            [
                False, False, False, # ACTIVE SUBENV
                False, False, True, True, # MASK
                False, # PAD
            ]
            # fmt: on
        ),
        # Intermediate, before activating subenv
        (
            "env_three_cubes_grids",
            [
                [-1, 0, [0, 0, 1], [1, 0, -1]],
                {
                    0: [1, 1],
                    1: [0.25, 0.15],
                },
            ],
            # fmt: off
            [
                False, False, False, # ACTIVE SUBENV
                False, False, True, True, # MASK
                False, # PAD
            ]
            # fmt: on
        ),
        # Intermediate, before subenv action
        (
            "env_three_cubes_grids",
            [
                [0, 1, [0, 0, 1], [1, 0, -1]],
                {
                    0: [1, 1],
                    1: [0.25, 0.15],
                },
            ],
            # fmt: off
            [
                True, False, False, # ACTIVE SUBENV
                False, False, False, # MASK
                False, False, # PAD
            ]
            # fmt: on
        ),
        # Intermediate, after subenv action
        (
            "env_three_cubes_grids",
            [
                [0, 0, [0, 0, 1], [1, 0, -1]],
                {
                    0: [2, 1],
                    1: [0.25, 0.15],
                },
            ],
            # fmt: off
            [
                False, False, False, # ACTIVE SUBENV
                False, True, True, True, # MASK
                False, # PAD
            ]
            # fmt: on
        ),
        # Intermediate, one subenv is done
        (
            "env_three_cubes_grids",
            [
                [-1, 0, [0, 1, 1], [1, 0, -1]],
                {
                    0: [2, 1],
                    1: [0.25, 0.15],
                },
            ],
            # fmt: off
            [
                False, False, False, # ACTIVE SUBENV
                False, True, True, True, # MASK
                False, # PAD
            ]
            # fmt: on
        ),
        # All subenvs are done
        (
            "env_three_cubes_grids",
            [
                [-1, 0, [1, 1, 1], [1, 0, -1]],
                {
                    0: [2, 1],
                    1: [0.25, 0.15],
                },
            ],
            # fmt: off
            [
                False, False, False, # ACTIVE SUBENV
                True, True, True, False, # MASK
                False, # PAD
            ]
            # fmt: on
        ),
    ],
)
def test__get_mask_invalid_actions_forward__returns_expected_without_setting_subenvs(
    env, state, mask_exp, request
):
    env = request.getfixturevalue(env)
    # Passing state
    mask = env.get_mask_invalid_actions_forward(state, done=False)
    assert mask == mask_exp
    # State from env
    env.set_state(state, done=False)
    mask = env.get_mask_invalid_actions_forward()
    assert mask == mask_exp


@pytest.mark.parametrize(
    "env, subenvs, state, mask_exp",
    [
        # Source, one env
        (
            "env_two_grids",
            (Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),),
            [
                [-1, 0, [0, 1], [0, -1]],
                {
                    0: [0, 0],
                },
            ],
            # fmt: off
            [
                False, False, # ACTIVE SUBENV
                True, True, True, # MASK
            ]
            # fmt: on
        ),
        # Source, two envs
        (
            "env_two_grids",
            (
                Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),
                Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),
            ),
            [
                [-1, 0, [0, 0], [0, 0]],
                {
                    0: [0, 0],
                    1: [0, 0],
                },
            ],
            # fmt: off
            [
                False, False, # ACTIVE SUBENV
                True, True, True, # MASK
            ]
            # fmt: on
        ),
        # Intermediate, two envs
        # Active 0, toggle flag 1
        (
            "env_two_grids",
            (
                Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),
                Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),
            ),
            [
                [0, 1, [0, 0], [0, 0]],
                {
                    0: [1, 2],
                    1: [0, 1],
                },
            ],
            # fmt: off
            [
                False, False, # ACTIVE SUBENV
                False, True, True, # MASK
            ]
            # fmt: on
        ),
        # Intermediate, two envs
        # Active 1, toggle flag 1
        (
            "env_two_grids",
            (
                Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),
                Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),
            ),
            [
                [1, 1, [0, 0], [0, 0]],
                {
                    0: [1, 2],
                    1: [0, 1],
                },
            ],
            # fmt: off
            [
                False, False, # ACTIVE SUBENV
                True, False, True, # MASK
            ]
            # fmt: on
        ),
        # Source
        (
            "env_three_cubes_grids",
            (
                Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),
                ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
            ),
            [
                [-1, 0, [0, 0, 1], [1, 0, -1]],
                {
                    0: [0, 0],
                    1: [-1, -1],
                },
            ],
            # fmt: off
            [
                False, False, False, # ACTIVE SUBENV
                True, True, True, True, # MASK
                False, # PAD
            ]
            # fmt: on
        ),
        # Intermediate, no active environment
        (
            "env_three_cubes_grids",
            (
                Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),
                ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
            ),
            [
                [-1, 0, [0, 0, 1], [1, 0, -1]],
                {
                    0: [1, 1],
                    1: [0.25, 0.15],
                },
            ],
            # fmt: off
            [
                False, False, False, # ACTIVE SUBENV
                False, False, True, True, # MASK
                False, # PAD
            ]
            # fmt: on
        ),
        # Intermediate, before subenv action
        # Active 0, toggle flag 1
        (
            "env_three_cubes_grids",
            (
                Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),
                ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
            ),
            [
                [0, 1, [0, 0, 1], [1, 0, -1]],
                {
                    0: [1, 1],
                    1: [0.25, 0.15],
                },
            ],
            # fmt: off
            [
                False, False, False, # ACTIVE SUBENV
                False, True, True, True, # MASK
                False, # PAD
            ]
            # fmt: on
        ),
        # Intermediate, after subenv action
        # Active 0, toggle flag 0
        (
            "env_three_cubes_grids",
            (
                Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),
                ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
            ),
            [
                [0, 0, [0, 0, 1], [1, 0, -1]],
                {
                    0: [2, 1],
                    1: [0.25, 0.15],
                },
            ],
            # fmt: off
            [
                True, False, False, # ACTIVE SUBENV
                False, False, True, # MASK
                False, False, # PAD
            ]
            # fmt: on
        ),
        # Intermediate, no active subenv, one subenv is source
        (
            "env_three_cubes_grids",
            (
                Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),
                ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
            ),
            [
                [-1, 0, [0, 0, 1], [1, 0, -1]],
                {
                    0: [0, 0],
                    1: [0.25, 0.15],
                },
            ],
            # fmt: off
            [
                False, False, False, # ACTIVE SUBENV
                True, False, True, True, # MASK
                False, # PAD
            ]
            # fmt: on
        ),
        # All subenvs are done
        (
            "env_three_cubes_grids",
            (
                Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),
                ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1),
            ),
            [
                [-1, 0, [1, 1, 1], [1, 0, -1]],
                {
                    0: [2, 1],
                    1: [0.25, 0.15],
                },
            ],
            # fmt: off
            [
                False, False, False, # ACTIVE SUBENV
                True, True, True, False, # MASK
                False, # PAD
            ]
            # fmt: on
        ),
    ],
)
def test__get_mask_invalid_actions_backward__returns_expected(
    env, subenvs, state, mask_exp, request
):
    env = request.getfixturevalue(env)
    env.set_subenvs(subenvs)
    # Passing state
    mask = env.get_mask_invalid_actions_backward(state, done=all(env._get_dones(state)))
    assert mask == mask_exp
    # State from env
    env.set_state(state, done=all(env._get_dones(state)))
    mask = env.get_mask_invalid_actions_backward()
    assert mask == mask_exp


@pytest.mark.parametrize(
    "env, state, mask_exp",
    [
        # Source, one env
        (
            "env_two_grids",
            [
                [-1, 0, [0, 1], [0, -1]],
                {
                    0: [0, 0],
                },
            ],
            # fmt: off
            [
                False, False, # ACTIVE SUBENV
                True, True, True, # MASK
            ]
            # fmt: on
        ),
        # Source, two envs
        (
            "env_two_grids",
            [
                [-1, 0, [0, 0], [0, 0]],
                {
                    0: [0, 0],
                    1: [0, 0],
                },
            ],
            # fmt: off
            [
                False, False, # ACTIVE SUBENV
                True, True, True, # MASK
            ]
            # fmt: on
        ),
        # Intermediate, two envs
        # Active 0, toggle flag 1
        (
            "env_two_grids",
            [
                [0, 1, [0, 0], [0, 0]],
                {
                    0: [1, 2],
                    1: [0, 1],
                },
            ],
            # fmt: off
            [
                False, False, # ACTIVE SUBENV
                False, True, True, # MASK
            ]
            # fmt: on
        ),
        # Intermediate, two envs
        # Active 1, toggle flag 1
        (
            "env_two_grids",
            [
                [1, 1, [0, 0], [0, 0]],
                {
                    0: [1, 2],
                    1: [0, 1],
                },
            ],
            # fmt: off
            [
                False, False, # ACTIVE SUBENV
                True, False, True, # MASK
            ]
            # fmt: on
        ),
        # Source
        (
            "env_three_cubes_grids",
            [
                [-1, 0, [0, 0, 1], [1, 0, -1]],
                {
                    0: [0, 0],
                    1: [-1, -1],
                },
            ],
            # fmt: off
            [
                False, False, False, # ACTIVE SUBENV
                True, True, True, True, # MASK
                False, # PAD
            ]
            # fmt: on
        ),
        # Intermediate, no active environment
        (
            "env_three_cubes_grids",
            [
                [-1, 0, [0, 0, 1], [1, 0, -1]],
                {
                    0: [1, 1],
                    1: [0.25, 0.15],
                },
            ],
            # fmt: off
            [
                False, False, False, # ACTIVE SUBENV
                False, False, True, True, # MASK
                False, # PAD
            ]
            # fmt: on
        ),
        # Intermediate, before subenv action
        # Active 0, toggle flag 1
        (
            "env_three_cubes_grids",
            [
                [0, 1, [0, 0, 1], [1, 0, -1]],
                {
                    0: [1, 1],
                    1: [0.25, 0.15],
                },
            ],
            # fmt: off
            [
                False, False, False, # ACTIVE SUBENV
                False, True, True, True, # MASK
                False, # PAD
            ]
            # fmt: on
        ),
        # Intermediate, after subenv action
        # Active 0, toggle flag 0
        (
            "env_three_cubes_grids",
            [
                [0, 0, [0, 0, 1], [1, 0, -1]],
                {
                    0: [2, 1],
                    1: [0.25, 0.15],
                },
            ],
            # fmt: off
            [
                True, False, False, # ACTIVE SUBENV
                False, False, True, # MASK
                False, False, # PAD
            ]
            # fmt: on
        ),
        # Intermediate, no active subenv, one subenv is source
        (
            "env_three_cubes_grids",
            [
                [-1, 0, [0, 0, 1], [1, 0, -1]],
                {
                    0: [0, 0],
                    1: [0.25, 0.15],
                },
            ],
            # fmt: off
            [
                False, False, False, # ACTIVE SUBENV
                True, False, True, True, # MASK
                False, # PAD
            ]
            # fmt: on
        ),
        # All subenvs are done
        (
            "env_three_cubes_grids",
            [
                [-1, 0, [1, 1, 1], [1, 0, -1]],
                {
                    0: [2, 1],
                    1: [0.25, 0.15],
                },
            ],
            # fmt: off
            [
                False, False, False, # ACTIVE SUBENV
                True, True, True, False, # MASK
                False, # PAD
            ]
            # fmt: on
        ),
    ],
)
def test__get_mask_invalid_actions_backward__returns_expected_without_setting_subenvs(
    env, state, mask_exp, request
):
    env = request.getfixturevalue(env)
    # Passing state
    mask = env.get_mask_invalid_actions_backward(state, done=all(env._get_dones(state)))
    assert mask == mask_exp
    # State from env
    env.set_state(state, done=all(env._get_dones(state)))
    mask = env.get_mask_invalid_actions_backward()
    assert mask == mask_exp


@pytest.mark.parametrize(
    "env, mask, idx_unique, mask_core",
    [
        (
            "env_two_grids",
            # fmt: off
            [
                False, False, # ACTIVE SUBENV
                False, True, True, # CORE MASK
            ],
            # fmt: on
            -1,
            [False, True, True],
        ),
        (
            "env_two_grids",
            # fmt: off
            [
                False, False, # ACTIVE SUBENV
                False, False, True, # CORE MASK
            ],
            # fmt: on
            -1,
            [False, False, True],
        ),
        (
            "env_two_grids",
            # fmt: off
            [
                True, False, # ACTIVE SUBENV
                True, False, False, # CORE MASK
            ],
            # fmt: on
            0,
            [True, False, False],
        ),
        (
            "env_two_grids",
            # fmt: off
            [
                False, True, # ACTIVE SUBENV
                False, False, False, # CORE MASK
            ],
            # fmt: on
            -1,
            [False, False, False],
        ),
        (
            "env_two_grids",
            # fmt: off
            [
                False, False, # ACTIVE SUBENV
                False, True, True, # CORE MASK
            ],
            # fmt: on
            -1,
            [False, True, True],
        ),
        (
            "env_two_grids",
            # fmt: off
            [
                False, False, # ACTIVE SUBENV
                True, True, True, # CORE MASK
            ],
            # fmt: on
            -1,
            [True, True, True],
        ),
        (
            "env_three_cubes_grids",
            # fmt: off
            [
                False, False, False, # ACTIVE SUBENV
                False, False, True, True, # CORE MASK
                False, # PAD
            ],
            # fmt: on
            -1,
            [False, False, True, True],
        ),
        (
            "env_three_cubes_grids",
            # fmt: off
            [
                False, False, False, # ACTIVE SUBENV
                False, False, True, True, # CORE MASK
                False, # PAD
            ],
            # fmt: on
            -1,
            [False, False, True, True],
        ),
        (
            "env_three_cubes_grids",
            # fmt: off
            [
                True, False, False, # ACTIVE SUBENV
                False, False, False, # CORE MASK
                False, False, # PAD
            ],
            # fmt: on
            1,
            [False, False, False],
        ),
        (
            "env_three_cubes_grids",
            # fmt: off
            [
                False, False, False, # ACTIVE SUBENV
                False, True, True, True, # CORE MASK
                False, # PAD
            ],
            # fmt: on
            -1,
            [False, True, True, True],
        ),
        (
            "env_three_cubes_grids",
            # fmt: off
            [
                True, False, False, # ACTIVE SUBENV
                False, False, True, # CORE MASK
                False, False, # PAD
            ],
            # fmt: on
            1,
            [False, False, True],
        ),
    ],
)
def test__extract_core_mask__returns_expected(
    env, mask, idx_unique, mask_core, request
):
    env = request.getfixturevalue(env)
    if isinstance(mask, list):
        assert mask_core == env._extract_core_mask(mask, idx_unique)
    else:
        assert torch.equal(mask_core, env._extract_core_mask(mask, idx_unique))


@pytest.mark.parametrize(
    "env, state_from, action, state_next_exp, valid_exp",
    [
        # From source, one env: activate subenv 0
        (
            "env_two_grids",
            [
                [-1, 0, [0, 1], [0, -1]],
                {
                    0: [0, 0],
                },
            ],
            (-1, 0, 0),
            [
                [0, 1, [0, 1], [0, -1]],
                {
                    0: [0, 0],
                },
            ],
            True,
        ),
        # From source -> activate grid: grid action
        (
            "env_two_grids",
            [
                [0, 1, [0, 1], [0, -1]],
                {
                    0: [0, 0],
                },
            ],
            (0, 1, 0),
            [
                [0, 0, [0, 1], [0, -1]],
                {
                    0: [1, 0],
                },
            ],
            True,
        ),
        # From source -> activate grid -> grid action: toggle subenv 0
        (
            "env_two_grids",
            [
                [0, 0, [0, 1], [0, -1]],
                {
                    0: [1, 0],
                },
            ],
            (-1, 0, 0),
            [
                [-1, 0, [0, 1], [0, -1]],
                {
                    0: [1, 0],
                },
            ],
            True,
        ),
        # From done subenv, global EOS
        (
            "env_two_grids",
            [
                [-1, 0, [1, 1], [0, -1]],
                {
                    0: [1, 0],
                },
            ],
            (-1, -1, -1),
            [
                [-1, 0, [1, 1], [0, -1]],
                {
                    0: [1, 0],
                },
            ],
            True,
        ),
        # Invalid: From done subenv, activate subenv 0
        (
            "env_two_grids",
            [
                [-1, 0, [1, 1], [0, -1]],
                {
                    0: [1, 0],
                },
            ],
            (-1, 0, 0),
            [
                [-1, 0, [1, 1], [0, -1]],
                {
                    0: [1, 0],
                },
            ],
            False,
        ),
        # From source: activate subenv 0
        (
            "env_three_cubes_grids",
            [
                [-1, 0, [0, 0, 1], [1, 0, -1]],
                {
                    0: [0, 0],
                    1: [-1, -1],
                },
            ],
            (-1, 0, 0, 0),
            [
                [0, 1, [0, 0, 1], [1, 0, -1]],
                {
                    0: [0, 0],
                    1: [-1, -1],
                },
            ],
            True,
        ),
        # From source -> toggle 0: subenv 0 action
        (
            "env_three_cubes_grids",
            [
                [0, 1, [0, 0, 1], [1, 0, -1]],
                {
                    0: [0, 0],
                    1: [-1, -1],
                },
            ],
            (1, 1, 0, 0),
            [
                [0, 0, [0, 0, 1], [1, 0, -1]],
                {
                    0: [1, 0],
                    1: [-1, -1],
                },
            ],
            True,
        ),
        # From source -> toggle 0 -> subenv 0 action: toggle 0
        (
            "env_three_cubes_grids",
            [
                [0, 0, [0, 0, 1], [1, 0, -1]],
                {
                    0: [1, 0],
                    1: [-1, -1],
                },
            ],
            (-1, 0, 0, 0),
            [
                [-1, 0, [0, 0, 1], [1, 0, -1]],
                {
                    0: [1, 0],
                    1: [-1, -1],
                },
            ],
            True,
        ),
        # Invalid: From source -> toggle 0 -> subenv 0 action: toggle 1
        (
            "env_three_cubes_grids",
            [
                [0, 0, [0, 0, 1], [1, 0, -1]],
                {
                    0: [1, 0],
                    1: [-1, -1],
                },
            ],
            (-1, 1, 0, 0),
            [
                [0, 0, [0, 0, 1], [1, 0, -1]],
                {
                    0: [1, 0],
                    1: [-1, -1],
                },
            ],
            False,
        ),
        # From intermediate: toggle 1
        (
            "env_three_cubes_grids",
            [
                [-1, 0, [0, 0, 1], [1, 0, -1]],
                {
                    0: [1, 0],
                    1: [-1, -1],
                },
            ],
            (-1, 1, 0, 0),
            [
                [1, 1, [0, 0, 1], [1, 0, -1]],
                {
                    0: [1, 0],
                    1: [-1, -1],
                },
            ],
            True,
        ),
        # From active 1, subenv 1 action
        (
            "env_three_cubes_grids",
            [
                [1, 1, [0, 0, 1], [1, 0, -1]],
                {
                    0: [1, 0],
                    1: [-1, -1],
                },
            ],
            (0, 0.25, 0.15, 1),
            [
                [1, 0, [0, 0, 1], [1, 0, -1]],
                {
                    0: [1, 0],
                    1: [0.25, 0.15],
                },
            ],
            True,
        ),
        # Invalid: From active 1, invalid subenv 1 action
        (
            "env_three_cubes_grids",
            [
                [1, 1, [0, 0, 1], [1, 0, -1]],
                {
                    0: [1, 0],
                    1: [-1, -1],
                },
            ],
            (0, 0.25, 0.15, 0),
            [
                [1, 1, [0, 0, 1], [1, 0, -1]],
                {
                    0: [1, 0],
                    1: [-1, -1],
                },
            ],
            False,
        ),
        # From active 1 -> subenv 1 action: toggle 1
        (
            "env_three_cubes_grids",
            [
                [1, 0, [0, 0, 1], [1, 0, -1]],
                {
                    0: [1, 0],
                    1: [0.25, 0.15],
                },
            ],
            (-1, 1, 0, 0),
            [
                [-1, 0, [0, 0, 1], [1, 0, -1]],
                {
                    0: [1, 0],
                    1: [0.25, 0.15],
                },
            ],
            True,
        ),
        # Invalid: From active 1 -> subenv 1 action: toggle 0
        (
            "env_three_cubes_grids",
            [
                [1, 0, [0, 0, 1], [1, 0, -1]],
                {
                    0: [1, 0],
                    1: [0.25, 0.15],
                },
            ],
            (-1, 0, 0, 0),
            [
                [1, 0, [0, 0, 1], [1, 0, -1]],
                {
                    0: [1, 0],
                    1: [0.25, 0.15],
                },
            ],
            False,
        ),
        # From all subenvs done, global EOS
        (
            "env_three_cubes_grids",
            [
                [-1, 0, [1, 1, 1], [1, 0, -1]],
                {
                    0: [1, 0],
                    1: [0.25, 0.15],
                },
            ],
            (-1, -1, -1, -1),
            [
                [-1, 0, [1, 1, 1], [1, 0, -1]],
                {
                    0: [1, 0],
                    1: [0.25, 0.15],
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
    warnings.filterwarnings("ignore")
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
            "env_two_grids",
            [
                [0, 1, [0, 1], [0, -1]],
                {
                    0: [0, 0],
                },
            ],
            (-1, 0, 0),
            [
                [-1, 0, [0, 1], [0, -1]],
                {
                    0: [0, 0],
                },
            ],
            True,
        ),
        (
            "env_two_grids",
            [
                [0, 0, [0, 1], [0, -1]],
                {
                    0: [1, 0],
                },
            ],
            (0, 1, 0),
            [
                [0, 1, [0, 1], [0, -1]],
                {
                    0: [0, 0],
                },
            ],
            True,
        ),
        (
            "env_two_grids",
            [
                [-1, 0, [0, 1], [0, -1]],
                {
                    0: [1, 0],
                },
            ],
            (-1, 0, 0),
            [
                [0, 0, [0, 1], [0, -1]],
                {
                    0: [1, 0],
                },
            ],
            True,
        ),
        (
            "env_two_grids",
            [
                [-1, 0, [1, 1], [0, -1]],
                {
                    0: [1, 0],
                },
            ],
            (-1, 0, 0),
            [
                [0, 0, [1, 1], [0, -1]],
                {
                    0: [1, 0],
                },
            ],
            True,
        ),
        (
            "env_three_cubes_grids",
            [
                [0, 1, [0, 0, 1], [1, 0, -1]],
                {
                    0: [0, 0],
                    1: [-1, -1],
                },
            ],
            (-1, 0, 0, 0),
            [
                [-1, 0, [0, 0, 1], [1, 0, -1]],
                {
                    0: [0, 0],
                    1: [-1, -1],
                },
            ],
            True,
        ),
        (
            "env_three_cubes_grids",
            [
                [0, 0, [0, 0, 1], [1, 0, -1]],
                {
                    0: [1, 0],
                    1: [-1, -1],
                },
            ],
            (1, 1, 0, 0),
            [
                [0, 1, [0, 0, 1], [1, 0, -1]],
                {
                    0: [0, 0],
                    1: [-1, -1],
                },
            ],
            True,
        ),
        (
            "env_three_cubes_grids",
            [
                [-1, 0, [0, 0, 1], [1, 0, -1]],
                {
                    0: [1, 0],
                    1: [-1, -1],
                },
            ],
            (-1, 0, 0, 0),
            [
                [0, 0, [0, 0, 1], [1, 0, -1]],
                {
                    0: [1, 0],
                    1: [-1, -1],
                },
            ],
            True,
        ),
        (
            "env_three_cubes_grids",
            [
                [0, 0, [0, 0, 1], [1, 0, -1]],
                {
                    0: [1, 0],
                    1: [-1, -1],
                },
            ],
            (-1, 1, 0, 0),
            [
                [0, 0, [0, 0, 1], [1, 0, -1]],
                {
                    0: [1, 0],
                    1: [-1, -1],
                },
            ],
            False,
        ),
        # From intermediate: toggle 1
        (
            "env_three_cubes_grids",
            [
                [1, 1, [0, 0, 1], [1, 0, -1]],
                {
                    0: [1, 0],
                    1: [-1, -1],
                },
            ],
            (-1, 1, 0, 0),
            [
                [-1, 0, [0, 0, 1], [1, 0, -1]],
                {
                    0: [1, 0],
                    1: [-1, -1],
                },
            ],
            True,
        ),
        # From active 1, subenv 1 action
        (
            "env_three_cubes_grids",
            [
                [1, 0, [0, 0, 1], [1, 0, -1]],
                {
                    0: [1, 0],
                    1: [0.25, 0.15],
                },
            ],
            (0, 0.25, 0.15, 1),
            [
                [1, 1, [0, 0, 1], [1, 0, -1]],
                {
                    0: [1, 0],
                    1: [-1, -1],
                },
            ],
            True,
        ),
        # Invalid: From active 1, invalid subenv 1 action
        (
            "env_three_cubes_grids",
            [
                [1, 1, [0, 0, 1], [1, 0, -1]],
                {
                    0: [1, 0],
                    1: [-1, -1],
                },
            ],
            (0, 0.25, 0.15, 0),
            [
                [1, 1, [0, 0, 1], [1, 0, -1]],
                {
                    0: [1, 0],
                    1: [-1, -1],
                },
            ],
            False,
        ),
        # From active 1 -> subenv 1 action: toggle 1
        (
            "env_three_cubes_grids",
            [
                [-1, 0, [0, 0, 1], [1, 0, -1]],
                {
                    0: [1, 0],
                    1: [0.25, 0.15],
                },
            ],
            (-1, 1, 0, 0),
            [
                [1, 0, [0, 0, 1], [1, 0, -1]],
                {
                    0: [1, 0],
                    1: [0.25, 0.15],
                },
            ],
            True,
        ),
        # Invalid: From active 1 -> subenv 1 action: toggle 0
        (
            "env_three_cubes_grids",
            [
                [1, 0, [0, 0, 1], [1, 0, -1]],
                {
                    0: [1, 0],
                    1: [0.25, 0.15],
                },
            ],
            (-1, 0, 0, 0),
            [
                [1, 0, [0, 0, 1], [1, 0, -1]],
                {
                    0: [1, 0],
                    1: [0.25, 0.15],
                },
            ],
            False,
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
    warnings.filterwarnings("ignore")
    state_next, action_done, valid = env.step_backwards(action)

    # Check end state
    assert env.equal(env.state, state_next)
    assert env.equal(env.state, state_next_exp)

    # Check action and valid
    assert action_done == action
    assert valid == valid_exp, (state_from, action)


@pytest.mark.parametrize(
    "env, state_from, action, state_next_exp, valid_exp",
    [
        (
            "env_two_grids",
            [
                [-1, 0, [1, 1], [0, -1]],
                {
                    0: [1, 0],
                },
            ],
            (-1, -1, -1),
            [
                [-1, 0, [1, 1], [0, -1]],
                {
                    0: [1, 0],
                },
            ],
            True,
        ),
        # Invalid: activate subenv 0
        (
            "env_two_grids",
            [
                [-1, 0, [1, 1], [0, -1]],
                {
                    0: [1, 0],
                },
            ],
            (-1, 0, 0),
            [
                [-1, 0, [1, 1], [0, -1]],
                {
                    0: [1, 0],
                },
            ],
            False,
        ),
        (
            "env_three_cubes_grids",
            [
                [-1, 0, [1, 1, 1], [1, 0, -1]],
                {
                    0: [1, 0],
                    1: [0.25, 0.15],
                },
            ],
            (-1, -1, -1, -1),
            [
                [-1, 0, [1, 1, 1], [1, 0, -1]],
                {
                    0: [1, 0],
                    1: [0.25, 0.15],
                },
            ],
            True,
        ),
        # Invalid: activate subenv 0
        (
            "env_three_cubes_grids",
            [
                [-1, 0, [1, 1, 1], [1, 0, -1]],
                {
                    0: [1, 0],
                    1: [0.25, 0.15],
                },
            ],
            (-1, 0, 0, 0),
            [
                [-1, 0, [1, 1, 1], [1, 0, -1]],
                {
                    0: [1, 0],
                    1: [0.25, 0.15],
                },
            ],
            False,
        ),
        # Invalid: activate subenv 1
        (
            "env_three_cubes_grids",
            [
                [-1, 0, [1, 1, 1], [1, 0, -1]],
                {
                    0: [1, 0],
                    1: [0.25, 0.15],
                },
            ],
            (-1, 1, 0, 0),
            [
                [-1, 0, [1, 1, 1], [1, 0, -1]],
                {
                    0: [1, 0],
                    1: [0.25, 0.15],
                },
            ],
            False,
        ),
    ],
)
def test__step_backwards_from_global_done__works_as_expected(
    env, state_from, action, state_next_exp, valid_exp, request
):
    env = request.getfixturevalue(env)
    env.set_state(state_from, done=True)

    # Check init state
    assert env.equal(env.state, state_from)

    # Perform step
    warnings.filterwarnings("ignore")
    state_next, action_done, valid = env.step_backwards(action)

    # Check end state
    assert env.equal(env.state, state_next)
    assert env.equal(env.state, state_next_exp)

    # Check action and valid
    assert action_done == action
    assert valid == valid_exp, (state_from, action)


@pytest.mark.parametrize(
    "env, state, valid_actions",
    [
        (
            "env_two_grids",
            # One subenv only
            # Source
            [
                [-1, 0, [0, 1], [0, -1]],
                {
                    0: [0, 0],
                },
            ],
            [(-1, 0, 0)],
        ),
        (
            "env_two_grids",
            # One subenv only
            # Active subenv 0, toggle flag 1
            [
                [0, 1, [0, 1], [0, -1]],
                {
                    0: [0, 0],
                },
            ],
            [(0, 0, 0), (0, 1, 0), (0, 0, 1)],
        ),
        (
            "env_two_grids",
            # One subenv only
            # Active subenv 0, toggle flag 0
            [
                [0, 0, [0, 1], [0, -1]],
                {
                    0: [0, 0],
                },
            ],
            [(-1, 0, 0)],
        ),
        (
            "env_two_grids",
            # One subenv only
            # Subenv done
            [
                [-1, 0, [1, 1], [0, -1]],
                {
                    0: [1, 1],
                },
            ],
            [(-1, -1, -1)],
        ),
        (
            "env_two_grids",
            # Two subenvs
            # Source
            [
                [-1, 0, [0, 0], [0, 0]],
                {
                    0: [0, 0],
                    1: [0, 0],
                },
            ],
            [(-1, 0, 0), (-1, 1, 0)],
        ),
        (
            "env_two_grids",
            # Two subenvs
            # Active subenv 0, toggle flag 1
            [
                [0, 1, [0, 0], [0, 0]],
                {
                    0: [0, 0],
                    1: [0, 0],
                },
            ],
            [(0, 0, 0), (0, 1, 0), (0, 0, 1)],
        ),
        (
            "env_two_grids",
            # Two subenvs
            # Active subenv 1, toggle flag 1
            [
                [1, 1, [0, 0], [0, 0]],
                {
                    0: [0, 0],
                    1: [0, 0],
                },
            ],
            [(0, 0, 0), (0, 1, 0), (0, 0, 1)],
        ),
        (
            "env_two_grids",
            # Two subenvs
            # Active subenv 0, toggle flag 0
            [
                [0, 0, [0, 0], [0, 0]],
                {
                    0: [0, 0],
                    1: [0, 0],
                },
            ],
            [(-1, 0, 0)],
        ),
        (
            "env_two_grids",
            # Two subenvs
            # Active subenv 1, toggle flag 0
            [
                [1, 0, [0, 0], [0, 0]],
                {
                    0: [0, 0],
                    1: [0, 0],
                },
            ],
            [(-1, 1, 0)],
        ),
        (
            "env_three_cubes_grids",
            # One subenv only
            # Source
            [
                [-1, 0, [0, 1, 1], [0, -1, -1]],
                {
                    0: [-1, -1],
                },
            ],
            [(-1, 0, 0, 0)],
        ),
        (
            "env_three_cubes_grids",
            # Two subenvs
            # Source
            [
                [-1, 0, [0, 0, 1], [0, 1, -1]],
                {
                    0: [-1, -1],
                    1: [0, 0],
                },
            ],
            [(-1, 0, 0, 0), (-1, 1, 0, 0)],
        ),
        (
            "env_three_cubes_grids",
            # Three subenvs
            # Source
            [
                [-1, 0, [0, 0, 0], [0, 1, 0]],
                {
                    0: [-1, -1],
                    1: [0, 0],
                    2: [-1, -1],
                },
            ],
            [(-1, 0, 0, 0), (-1, 1, 0, 0), (-1, 2, 0, 0)],
        ),
        (
            "env_three_cubes_grids",
            # Three subenvs
            # Source
            [
                [-1, 0, [0, 0, 0], [1, 0, 1]],
                {
                    0: [0, 0],
                    1: [-1, -1],
                    2: [0, 0],
                },
            ],
            [(-1, 0, 0, 0), (-1, 1, 0, 0), (-1, 2, 0, 0)],
        ),
        (
            "env_three_cubes_grids",
            # Three subenvs
            # Active subenv 1, toggle 1
            [
                [1, 1, [0, 0, 0], [0, 1, 0]],
                {
                    0: [-1, -1],
                    1: [0, 0],
                    2: [-1, -1],
                },
            ],
            [(1, 0, 0, 0), (1, 1, 0, 0), (1, 0, 1, 0)],
        ),
        (
            "env_three_cubes_grids",
            # Three subenvs
            # Active subenv 1, toggle 0
            [
                [1, 0, [0, 0, 0], [0, 1, 0]],
                {
                    0: [-1, -1],
                    1: [0, 0],
                    2: [-1, -1],
                },
            ],
            [(-1, 1, 0, 0)],
        ),
        (
            "env_three_cubes_grids",
            # Three subenvs
            # Active subenv 2, toggle 1
            [
                [2, 1, [0, 0, 0], [0, 1, 0]],
                {
                    0: [-1, -1],
                    1: [0, 0],
                    2: [-1, -1],
                },
            ],
            [(0, 0.0, 0.0, 1)],
        ),
        (
            "env_three_cubes_grids",
            # Three subenvs
            # Active subenv 1, toggle 0
            [
                [2, 0, [0, 0, 0], [0, 1, 0]],
                {
                    0: [-1, -1],
                    1: [0, 0],
                    2: [-1, -1],
                },
            ],
            [(-1, 2, 0, 0)],
        ),
        (
            "env_three_cubes_grids",
            # Three subenvs
            # No active subenvs, with done subenvs
            [
                [-1, 0, [1, 1, 0], [1, 0, 1]],
                {
                    0: [1, 2],
                    1: [0.25, 0.75],
                    2: [0, 0],
                },
            ],
            [(-1, 2, 0, 0)],
        ),
    ],
)
def test__get_valid_actions__forward_returns_expected(
    env, state, valid_actions, request
):
    env = request.getfixturevalue(env)
    # Check the method by passing the state
    assert set(valid_actions) == set(
        env.get_valid_actions(state=state, done=False, backward=False)
    )
    # Check the method after setting the state
    env.set_state(state, done=False)
    assert set(valid_actions) == set(env.get_valid_actions(backward=False))


@pytest.mark.repeat(10)
@pytest.mark.parametrize(
    "env, states",
    [
        (
            "env_two_grids",
            # Same elements in the sets
            # Two source states
            [
                [
                    [-1, 0, [0, 1], [0, -1]],
                    {
                        0: [0, 0],
                    },
                ],
                [
                    [-1, 0, [0, 1], [0, -1]],
                    {
                        0: [0, 0],
                    },
                ],
            ],
        ),
        (
            "env_two_grids",
            # Mixed sets in the batch
            # Two source states
            [
                [
                    [-1, 0, [0, 1], [0, -1]],
                    {
                        0: [0, 0],
                    },
                ],
                [
                    [-1, 0, [0, 0], [0, 0]],
                    {
                        0: [0, 0],
                        1: [0, 0],
                    },
                ],
            ],
        ),
        (
            "env_two_grids",
            # Mixed sets in the batch
            # Multiple states
            [
                [
                    [-1, 0, [0, 1], [0, -1]],
                    {
                        0: [0, 0],
                    },
                ],
                [
                    [-1, 0, [0, 1], [0, -1]],
                    {
                        0: [1, 2],
                    },
                ],
                [
                    [0, 1, [0, 1], [0, -1]],
                    {
                        0: [1, 2],
                    },
                ],
                [
                    [0, 0, [0, 1], [0, -1]],
                    {
                        0: [1, 2],
                    },
                ],
                [
                    [-1, 0, [1, 1], [0, -1]],
                    {
                        0: [1, 2],
                    },
                ],
                [
                    [-1, 0, [0, 0], [0, 0]],
                    {
                        0: [0, 0],
                        1: [0, 0],
                    },
                ],
                [
                    [-1, 0, [0, 0], [0, 0]],
                    {
                        0: [1, 0],
                        1: [0, 1],
                    },
                ],
                [
                    [1, 0, [0, 0], [0, 0]],
                    {
                        0: [1, 0],
                        1: [0, 1],
                    },
                ],
                [
                    [1, 1, [0, 0], [0, 0]],
                    {
                        0: [1, 0],
                        1: [0, 1],
                    },
                ],
                [
                    [-1, 0, [1, 0], [0, 0]],
                    {
                        0: [1, 0],
                        1: [0, 1],
                    },
                ],
            ],
        ),
        (
            "env_three_cubes_grids",
            # Same elements in the sets
            # Two source states
            [
                [
                    [-1, 0, [0, 1, 1], [1, -1, -1]],
                    {
                        0: [0, 0],
                    },
                ],
                [
                    [-1, 0, [0, 1, 1], [1, -1, -1]],
                    {
                        0: [0, 0],
                    },
                ],
            ],
        ),
        (
            "env_three_cubes_grids",
            # Same elements in the sets
            # Two source states
            [
                [
                    [-1, 0, [0, 0, 1], [1, 1, -1]],
                    {
                        0: [0, 0],
                        1: [0, 0],
                    },
                ],
                [
                    [-1, 0, [0, 0, 1], [1, 1, -1]],
                    {
                        0: [0, 0],
                        1: [0, 0],
                    },
                ],
            ],
        ),
        (
            "env_three_cubes_grids",
            # Mixed sets in the batch
            # Two source states
            [
                [
                    [-1, 0, [0, 1, 1], [1, -1, -1]],
                    {
                        0: [0, 0],
                    },
                ],
                [
                    [-1, 0, [0, 0, 1], [1, 1, -1]],
                    {
                        0: [0, 0],
                        1: [0, 0],
                    },
                ],
            ],
        ),
        (
            "env_three_cubes_grids",
            # Mixed sets in the batch
            # Two source states
            [
                [
                    [-1, 0, [0, 1, 1], [1, -1, -1]],
                    {
                        0: [0, 0],
                    },
                ],
                [
                    [-1, 0, [0, 0, 1], [1, 1, -1]],
                    {
                        0: [0, 0],
                        1: [0, 0],
                    },
                ],
                [
                    [-1, 0, [0, 0, 0], [1, 1, 1]],
                    {
                        0: [0, 0],
                        1: [0, 0],
                        2: [0, 0],
                    },
                ],
                [
                    [-1, 0, [0, 1, 1], [0, -1, -1]],
                    {
                        0: [-1, -1],
                    },
                ],
                [
                    [-1, 0, [0, 0, 1], [0, 1, -1]],
                    {
                        0: [-1, -1],
                        1: [0, 0],
                    },
                ],
                [
                    [-1, 0, [0, 0, 0], [1, 0, 1]],
                    {
                        0: [0, 0],
                        1: [-1, -1],
                        2: [0, 0],
                    },
                ],
            ],
        ),
        (
            "env_three_cubes_grids",
            # Mixed sets in the batch
            # Multiple states
            [
                [
                    [-1, 0, [0, 1, 1], [1, -1, -1]],
                    {
                        0: [0, 0],
                    },
                ],
                [
                    [-1, 0, [0, 0, 1], [1, 1, -1]],
                    {
                        0: [0, 0],
                        1: [0, 0],
                    },
                ],
                [
                    [-1, 0, [0, 0, 0], [1, 1, 1]],
                    {
                        0: [0, 0],
                        1: [0, 0],
                        2: [0, 0],
                    },
                ],
                [
                    [-1, 0, [0, 1, 1], [0, -1, -1]],
                    {
                        0: [-1, -1],
                    },
                ],
                [
                    [-1, 0, [0, 0, 1], [0, 1, -1]],
                    {
                        0: [-1, -1],
                        1: [0, 0],
                    },
                ],
                [
                    [-1, 0, [0, 0, 0], [1, 0, 1]],
                    {
                        0: [0, 0],
                        1: [-1, -1],
                        2: [0, 0],
                    },
                ],
                [
                    [0, 1, [0, 1, 1], [1, -1, -1]],
                    {
                        0: [1, 2],
                    },
                ],
                [
                    [1, 0, [0, 0, 1], [1, 1, -1]],
                    {
                        0: [1, 1],
                        1: [2, 1],
                    },
                ],
                [
                    [-1, 0, [0, 0, 1], [1, 1, 1]],
                    {
                        0: [1, 1],
                        1: [0, 1],
                        2: [2, 2],
                    },
                ],
                [
                    [0, 1, [0, 1, 1], [0, -1, -1]],
                    {
                        0: [-1, -1],
                    },
                ],
                [
                    [-1, 0, [0, 0, 1], [0, 1, -1]],
                    {
                        0: [0.25, 0.15],
                        1: [1, 2],
                    },
                ],
                [
                    [-1, 0, [0, 0, 1], [1, 0, 1]],
                    {
                        0: [1, 1],
                        1: [0.1234, 0.4321],
                        2: [2, 0],
                    },
                ],
                [
                    [1, 1, [0, 0, 1], [1, 0, 1]],
                    {
                        0: [1, 1],
                        1: [0.1234, 0.4321],
                        2: [2, 0],
                    },
                ],
                [
                    [1, 1, [0, 0, 1], [0, 1, 0]],
                    {
                        0: [0.8765, 0.6543],
                        1: [2, 0],
                        2: [0.1234, 0.4321],
                    },
                ],
                [
                    [1, 0, [0, 0, 1], [0, 1, 0]],
                    {
                        0: [0.8765, 0.6543],
                        1: [2, 1],
                        2: [0.1234, 0.4321],
                    },
                ],
                [
                    [-1, 0, [0, 0, 0], [0, 1, 0]],
                    {
                        0: [0.8765, 0.6543],
                        1: [2, 1],
                        2: [0.1234, 0.4321],
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
        assert env.action2representative(action) in env.get_valid_actions(
            state=state, done=False, backward=False
        )


@pytest.mark.repeat(10)
@pytest.mark.parametrize(
    "env, states",
    [
        (
            "env_two_grids",
            # Mixed sets in the batch
            # Multiple states
            [
                [
                    [0, 1, [0, 1], [0, -1]],
                    {
                        0: [0, 0],
                    },
                ],
                [
                    [-1, 0, [0, 1], [0, -1]],
                    {
                        0: [1, 2],
                    },
                ],
                [
                    [0, 1, [0, 1], [0, -1]],
                    {
                        0: [1, 2],
                    },
                ],
                [
                    [0, 0, [0, 1], [0, -1]],
                    {
                        0: [1, 2],
                    },
                ],
                [
                    [-1, 0, [1, 1], [0, -1]],
                    {
                        0: [1, 2],
                    },
                ],
                [
                    [0, 0, [0, 0], [0, 0]],
                    {
                        0: [0, 1],
                        1: [0, 0],
                    },
                ],
                [
                    [-1, 0, [0, 0], [0, 0]],
                    {
                        0: [1, 0],
                        1: [0, 1],
                    },
                ],
                [
                    [1, 0, [0, 0], [0, 0]],
                    {
                        0: [1, 0],
                        1: [0, 1],
                    },
                ],
                [
                    [1, 1, [0, 0], [0, 0]],
                    {
                        0: [1, 0],
                        1: [0, 1],
                    },
                ],
                [
                    [-1, 0, [1, 0], [0, 0]],
                    {
                        0: [1, 0],
                        1: [0, 1],
                    },
                ],
            ],
        ),
        (
            "env_three_cubes_grids",
            # Mixed sets in the batch
            # Multiple states
            [
                [
                    [0, 1, [0, 1, 1], [1, -1, -1]],
                    {
                        0: [0, 0],
                    },
                ],
                [
                    [0, 1, [0, 0, 1], [1, 1, -1]],
                    {
                        0: [0, 0],
                        1: [0, 0],
                    },
                ],
                [
                    [1, 1, [0, 0, 0], [1, 1, 1]],
                    {
                        0: [0, 0],
                        1: [0, 1],
                        2: [0, 0],
                    },
                ],
                [
                    [0, 1, [0, 1, 1], [0, -1, -1]],
                    {
                        0: [-1, -1],
                    },
                ],
                [
                    [0, 0, [0, 0, 1], [0, 1, -1]],
                    {
                        0: [0.1234, 0.4321],
                        1: [0, 0],
                    },
                ],
                [
                    [1, 1, [0, 0, 0], [1, 0, 1]],
                    {
                        0: [0, 0],
                        1: [-1, -1],
                        2: [0, 0],
                    },
                ],
                [
                    [0, 1, [0, 1, 1], [1, -1, -1]],
                    {
                        0: [1, 2],
                    },
                ],
                [
                    [1, 0, [0, 0, 1], [1, 1, -1]],
                    {
                        0: [1, 1],
                        1: [2, 1],
                    },
                ],
                [
                    [-1, 0, [0, 0, 1], [1, 1, 1]],
                    {
                        0: [1, 1],
                        1: [0, 1],
                        2: [2, 2],
                    },
                ],
                [
                    [0, 1, [0, 1, 1], [0, -1, -1]],
                    {
                        0: [-1, -1],
                    },
                ],
                [
                    [-1, 0, [0, 0, 1], [0, 1, -1]],
                    {
                        0: [0.25, 0.15],
                        1: [1, 2],
                    },
                ],
                [
                    [-1, 0, [0, 0, 1], [1, 0, 1]],
                    {
                        0: [1, 1],
                        1: [0.1234, 0.4321],
                        2: [2, 0],
                    },
                ],
                [
                    [1, 1, [0, 0, 1], [1, 0, 1]],
                    {
                        0: [1, 1],
                        1: [0.1234, 0.4321],
                        2: [2, 0],
                    },
                ],
                [
                    [1, 1, [0, 0, 1], [0, 1, 0]],
                    {
                        0: [0.8765, 0.6543],
                        1: [2, 0],
                        2: [0.1234, 0.4321],
                    },
                ],
                [
                    [1, 0, [0, 0, 1], [0, 1, 0]],
                    {
                        0: [0.8765, 0.6543],
                        1: [2, 1],
                        2: [0.1234, 0.4321],
                    },
                ],
                [
                    [-1, 0, [0, 0, 0], [0, 1, 0]],
                    {
                        0: [0.8765, 0.6543],
                        1: [2, 1],
                        2: [0.1234, 0.4321],
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
        assert env.action2representative(action) in env.get_valid_actions(
            state=state, done=False, backward=True
        )


@pytest.mark.repeat(10)
@pytest.mark.parametrize(
    "env, states",
    [
        (
            "env_two_grids",
            # Same elements in the sets
            # Two source states
            [
                [
                    [-1, 0, [0, 1], [0, -1]],
                    {
                        0: [0, 0],
                    },
                ],
                [
                    [-1, 0, [0, 1], [0, -1]],
                    {
                        0: [0, 0],
                    },
                ],
            ],
        ),
        (
            "env_two_grids",
            # Mixed sets in the batch
            # Two source states
            [
                [
                    [-1, 0, [0, 1], [0, -1]],
                    {
                        0: [0, 0],
                    },
                ],
                [
                    [-1, 0, [0, 0], [0, 0]],
                    {
                        0: [0, 0],
                        1: [0, 0],
                    },
                ],
            ],
        ),
        (
            "env_two_grids",
            # Mixed sets in the batch
            # Multiple states
            [
                [
                    [-1, 0, [0, 1], [0, -1]],
                    {
                        0: [0, 0],
                    },
                ],
                [
                    [-1, 0, [0, 1], [0, -1]],
                    {
                        0: [1, 2],
                    },
                ],
                [
                    [0, 1, [0, 1], [0, -1]],
                    {
                        0: [1, 2],
                    },
                ],
                [
                    [0, 0, [0, 1], [0, -1]],
                    {
                        0: [1, 2],
                    },
                ],
                [
                    [-1, 0, [1, 1], [0, -1]],
                    {
                        0: [1, 2],
                    },
                ],
                [
                    [-1, 0, [0, 0], [0, 0]],
                    {
                        0: [0, 0],
                        1: [0, 0],
                    },
                ],
                [
                    [-1, 0, [0, 0], [0, 0]],
                    {
                        0: [1, 0],
                        1: [0, 1],
                    },
                ],
                [
                    [1, 0, [0, 0], [0, 0]],
                    {
                        0: [1, 0],
                        1: [0, 1],
                    },
                ],
                [
                    [1, 1, [0, 0], [0, 0]],
                    {
                        0: [1, 0],
                        1: [0, 1],
                    },
                ],
                [
                    [-1, 0, [1, 0], [0, 0]],
                    {
                        0: [1, 0],
                        1: [0, 1],
                    },
                ],
            ],
        ),
        (
            "env_three_cubes_grids",
            # Same elements in the sets
            # Two source states
            [
                [
                    [-1, 0, [0, 1, 1], [1, -1, -1]],
                    {
                        0: [0, 0],
                    },
                ],
                [
                    [-1, 0, [0, 1, 1], [1, -1, -1]],
                    {
                        0: [0, 0],
                    },
                ],
            ],
        ),
        (
            "env_three_cubes_grids",
            # Same elements in the sets
            # Two source states
            [
                [
                    [-1, 0, [0, 0, 1], [1, 1, -1]],
                    {
                        0: [0, 0],
                        1: [0, 0],
                    },
                ],
                [
                    [-1, 0, [0, 0, 1], [1, 1, -1]],
                    {
                        0: [0, 0],
                        1: [0, 0],
                    },
                ],
            ],
        ),
        (
            "env_three_cubes_grids",
            # Mixed sets in the batch
            # Two source states
            [
                [
                    [-1, 0, [0, 1, 1], [1, -1, -1]],
                    {
                        0: [0, 0],
                    },
                ],
                [
                    [-1, 0, [0, 0, 1], [1, 1, -1]],
                    {
                        0: [0, 0],
                        1: [0, 0],
                    },
                ],
            ],
        ),
        (
            "env_three_cubes_grids",
            # Mixed sets in the batch
            # Two source states
            [
                [
                    [-1, 0, [0, 1, 1], [1, -1, -1]],
                    {
                        0: [0, 0],
                    },
                ],
                [
                    [-1, 0, [0, 0, 1], [1, 1, -1]],
                    {
                        0: [0, 0],
                        1: [0, 0],
                    },
                ],
                [
                    [-1, 0, [0, 0, 0], [1, 1, 1]],
                    {
                        0: [0, 0],
                        1: [0, 0],
                        2: [0, 0],
                    },
                ],
                [
                    [-1, 0, [0, 1, 1], [0, -1, -1]],
                    {
                        0: [-1, -1],
                    },
                ],
                [
                    [-1, 0, [0, 0, 1], [0, 1, -1]],
                    {
                        0: [-1, -1],
                        1: [0, 0],
                    },
                ],
                [
                    [-1, 0, [0, 0, 0], [1, 0, 1]],
                    {
                        0: [0, 0],
                        1: [-1, -1],
                        2: [0, 0],
                    },
                ],
            ],
        ),
        (
            "env_three_cubes_grids",
            # Mixed sets in the batch
            # Multiple states
            [
                [
                    [-1, 0, [0, 1, 1], [1, -1, -1]],
                    {
                        0: [0, 0],
                    },
                ],
                [
                    [-1, 0, [0, 0, 1], [1, 1, -1]],
                    {
                        0: [0, 0],
                        1: [0, 0],
                    },
                ],
                [
                    [-1, 0, [0, 0, 0], [1, 1, 1]],
                    {
                        0: [0, 0],
                        1: [0, 0],
                        2: [0, 0],
                    },
                ],
                [
                    [-1, 0, [0, 1, 1], [0, -1, -1]],
                    {
                        0: [-1, -1],
                    },
                ],
                [
                    [-1, 0, [0, 0, 1], [0, 1, -1]],
                    {
                        0: [-1, -1],
                        1: [0, 0],
                    },
                ],
                [
                    [-1, 0, [0, 0, 0], [1, 0, 1]],
                    {
                        0: [0, 0],
                        1: [-1, -1],
                        2: [0, 0],
                    },
                ],
                [
                    [0, 1, [0, 1, 1], [1, -1, -1]],
                    {
                        0: [1, 2],
                    },
                ],
                [
                    [1, 0, [0, 0, 1], [1, 1, -1]],
                    {
                        0: [1, 1],
                        1: [2, 1],
                    },
                ],
                [
                    [-1, 0, [0, 0, 1], [1, 1, 1]],
                    {
                        0: [1, 1],
                        1: [0, 1],
                        2: [2, 2],
                    },
                ],
                [
                    [0, 1, [0, 1, 1], [0, -1, -1]],
                    {
                        0: [-1, -1],
                    },
                ],
                [
                    [-1, 0, [0, 0, 1], [0, 1, -1]],
                    {
                        0: [0.25, 0.15],
                        1: [1, 2],
                    },
                ],
                [
                    [-1, 0, [0, 0, 1], [1, 0, 1]],
                    {
                        0: [1, 1],
                        1: [0.1234, 0.4321],
                        2: [2, 0],
                    },
                ],
                [
                    [1, 1, [0, 0, 1], [1, 0, 1]],
                    {
                        0: [1, 1],
                        1: [0.1234, 0.4321],
                        2: [2, 0],
                    },
                ],
                [
                    [1, 1, [0, 0, 1], [0, 1, 0]],
                    {
                        0: [0.8765, 0.6543],
                        1: [2, 0],
                        2: [0.1234, 0.4321],
                    },
                ],
                [
                    [1, 0, [0, 0, 1], [0, 1, 0]],
                    {
                        0: [0.8765, 0.6543],
                        1: [2, 1],
                        2: [0.1234, 0.4321],
                    },
                ],
                [
                    [-1, 0, [0, 0, 0], [0, 1, 0]],
                    {
                        0: [0.8765, 0.6543],
                        1: [2, 1],
                        2: [0.1234, 0.4321],
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


@pytest.mark.parametrize(
    "env, states, actions, policy_outputs",
    [
        (
            "env_three_cubes_grids",
            # Found while debugging
            [
                [[-1, 0, [0, 0, 1], [0, 0, -1]], {0: [-1, -1], 1: [-1, -1]}],
                [[-1, 0, [0, 0, 1], [0, 0, -1]], {0: [-1, -1], 1: [-1, -1]}],
            ],
            [(-1, 1, 0, 0), (-1, 0, 0, 0)],
            [
                [
                    -0.0647,
                    -0.0272,
                    -0.1068,
                    0.0577,
                    0.0528,
                    -0.0075,
                    0.0351,
                    -0.0608,
                    -0.0014,
                    -0.0209,
                    -0.0868,
                    0.1798,
                    0.1012,
                    0.0488,
                    0.1486,
                    0.0144,
                    -0.0078,
                    0.0414,
                    -0.0443,
                    -0.0765,
                    -0.0751,
                    -0.0947,
                    0.0541,
                    0.1114,
                    -0.0570,
                    0.0450,
                    -0.1312,
                ],
                [
                    -0.0647,
                    -0.0272,
                    -0.1068,
                    0.0577,
                    0.0528,
                    -0.0075,
                    0.0351,
                    -0.0608,
                    -0.0014,
                    -0.0209,
                    -0.0868,
                    0.1798,
                    0.1012,
                    0.0488,
                    0.1486,
                    0.0144,
                    -0.0078,
                    0.0414,
                    -0.0443,
                    -0.0765,
                    -0.0751,
                    -0.0947,
                    0.0541,
                    0.1114,
                    -0.0570,
                    0.0450,
                    -0.1312,
                ],
            ],
        ),
    ],
)
def test__get_logprobs_forward_with_actions_policy_outputs__all_finite(
    env, states, actions, policy_outputs, request
):
    env = request.getfixturevalue(env)
    n_states = len(states)
    # Get masks
    masks = tbool(
        [env.get_mask_invalid_actions_forward(s) for s in states], device=env.device
    )
    # Build policy outputs
    policy_outputs = tfloat(policy_outputs, float_type=env.float, device=env.device)
    # Build actions tensor
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
            "env_two_grids",
            # Mixed sets in the batch
            # Multiple states
            [
                [
                    [0, 1, [0, 1], [0, -1]],
                    {
                        0: [0, 0],
                    },
                ],
                [
                    [-1, 0, [0, 1], [0, -1]],
                    {
                        0: [1, 2],
                    },
                ],
                [
                    [0, 1, [0, 1], [0, -1]],
                    {
                        0: [1, 2],
                    },
                ],
                [
                    [0, 0, [0, 1], [0, -1]],
                    {
                        0: [1, 2],
                    },
                ],
                [
                    [-1, 0, [1, 1], [0, -1]],
                    {
                        0: [1, 2],
                    },
                ],
                [
                    [0, 0, [0, 0], [0, 0]],
                    {
                        0: [0, 1],
                        1: [0, 0],
                    },
                ],
                [
                    [-1, 0, [0, 0], [0, 0]],
                    {
                        0: [1, 0],
                        1: [0, 1],
                    },
                ],
                [
                    [1, 0, [0, 0], [0, 0]],
                    {
                        0: [1, 0],
                        1: [0, 1],
                    },
                ],
                [
                    [1, 1, [0, 0], [0, 0]],
                    {
                        0: [1, 0],
                        1: [0, 1],
                    },
                ],
                [
                    [-1, 0, [1, 0], [0, 0]],
                    {
                        0: [1, 0],
                        1: [0, 1],
                    },
                ],
            ],
        ),
        (
            "env_three_cubes_grids",
            # Mixed sets in the batch
            # Multiple states
            [
                [
                    [0, 1, [0, 1, 1], [1, -1, -1]],
                    {
                        0: [0, 0],
                    },
                ],
                [
                    [0, 1, [0, 0, 1], [1, 1, -1]],
                    {
                        0: [0, 0],
                        1: [0, 0],
                    },
                ],
                [
                    [1, 1, [0, 0, 0], [1, 1, 1]],
                    {
                        0: [0, 0],
                        1: [0, 1],
                        2: [0, 0],
                    },
                ],
                [
                    [0, 1, [0, 1, 1], [0, -1, -1]],
                    {
                        0: [-1, -1],
                    },
                ],
                [
                    [0, 0, [0, 0, 1], [0, 1, -1]],
                    {
                        0: [0.1234, 0.4321],
                        1: [0, 0],
                    },
                ],
                [
                    [1, 1, [0, 0, 0], [1, 0, 1]],
                    {
                        0: [0, 0],
                        1: [-1, -1],
                        2: [0, 0],
                    },
                ],
                [
                    [0, 1, [0, 1, 1], [1, -1, -1]],
                    {
                        0: [1, 2],
                    },
                ],
                [
                    [1, 0, [0, 0, 1], [1, 1, -1]],
                    {
                        0: [1, 1],
                        1: [2, 1],
                    },
                ],
                [
                    [-1, 0, [0, 0, 1], [1, 1, 1]],
                    {
                        0: [1, 1],
                        1: [0, 1],
                        2: [2, 2],
                    },
                ],
                [
                    [0, 1, [0, 1, 1], [0, -1, -1]],
                    {
                        0: [-1, -1],
                    },
                ],
                [
                    [-1, 0, [0, 0, 1], [0, 1, -1]],
                    {
                        0: [0.25, 0.15],
                        1: [1, 2],
                    },
                ],
                [
                    [-1, 0, [0, 0, 1], [1, 0, 1]],
                    {
                        0: [1, 1],
                        1: [0.1234, 0.4321],
                        2: [2, 0],
                    },
                ],
                [
                    [1, 1, [0, 0, 1], [1, 0, 1]],
                    {
                        0: [1, 1],
                        1: [0.1234, 0.4321],
                        2: [2, 0],
                    },
                ],
                [
                    [1, 1, [0, 0, 1], [0, 1, 0]],
                    {
                        0: [0.8765, 0.6543],
                        1: [2, 0],
                        2: [0.1234, 0.4321],
                    },
                ],
                [
                    [1, 0, [0, 0, 1], [0, 1, 0]],
                    {
                        0: [0.8765, 0.6543],
                        1: [2, 1],
                        2: [0.1234, 0.4321],
                    },
                ],
                [
                    [-1, 0, [0, 0, 0], [0, 1, 0]],
                    {
                        0: [0.8765, 0.6543],
                        1: [2, 1],
                        2: [0.1234, 0.4321],
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
    "env, states, states_policy_exp",
    [
        (
            "env_two_grids",
            [
                [[-1, 0, [0, 0], [0, -1]], {0: [0, 0]}],
                [[-1, 0, [0, 0], [0, 0]], {0: [0, 0], 1: [0, 0]}],
            ],
            torch.stack(
                [
                    torch.tensor(
                        [
                            # fmt: off
                            0.0, 0.0, # ACTIVE SUBENV
                            0.0, # TOGGLE FLAG
                            0.0, 0.0, # DONES
                            1.0, # PRESENT SUBENVS OF EACH TYPE
                            # GRID 0
                            1.0, 0.0, 0.0,
                            1.0, 0.0, 0.0,
                            # GRID 1
                            1.0, 0.0, 0.0,
                            1.0, 0.0, 0.0,
                            # fmt: on
                        ],
                        dtype=torch.float,
                    ),
                    torch.tensor(
                        [
                            # fmt: off
                            0.0, 0.0, # ACTIVE SUBENV
                            0.0, # TOGGLE FLAG
                            0.0, 0.0, # DONES
                            2.0, # PRESENT SUBENVS OF EACH TYPE
                            # GRID 0
                            1.0, 0.0, 0.0,
                            1.0, 0.0, 0.0,
                            # GRID 1
                            1.0, 0.0, 0.0,
                            1.0, 0.0, 0.0,
                            # fmt: on
                        ],
                        dtype=torch.float,
                    ),
                ],
                dim=0,
            ),
        ),
        (
            "env_two_grids",
            [
                [[-1, 0, [0, 0], [0, -1]], {0: [0, 0]}],
                [[-1, 0, [0, 0], [0, 0]], {0: [0, 0], 1: [0, 0]}],
                [[0, 1, [0, 0], [0, 0]], {0: [1, 1], 1: [0, 1]}],
            ],
            torch.stack(
                [
                    torch.tensor(
                        [
                            # fmt: off
                            0.0, 0.0, # ACTIVE SUBENV
                            0.0, # TOGGLE FLAG
                            0.0, 0.0, # DONES
                            1.0, # PRESENT SUBENVS OF EACH TYPE
                            # GRID 0
                            1.0, 0.0, 0.0,
                            1.0, 0.0, 0.0,
                            # GRID 1
                            1.0, 0.0, 0.0,
                            1.0, 0.0, 0.0,
                            # fmt: on
                        ],
                        dtype=torch.float,
                    ),
                    torch.tensor(
                        [
                            # fmt: off
                            0.0, 0.0, # ACTIVE SUBENV
                            0.0, # TOGGLE FLAG
                            0.0, 0.0, # DONES
                            2.0, # PRESENT SUBENVS OF EACH TYPE
                            # GRID 0
                            1.0, 0.0, 0.0,
                            1.0, 0.0, 0.0,
                            # GRID 1
                            1.0, 0.0, 0.0,
                            1.0, 0.0, 0.0,
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
                            2.0, # PRESENT SUBENVS OF EACH TYPE
                            # GRID 0
                            0.0, 1.0, 0.0,
                            0.0, 1.0, 0.0,
                            # GRID 1
                            1.0, 0.0, 0.0,
                            0.0, 1.0, 0.0,
                            # fmt: on
                        ],
                        dtype=torch.float,
                    ),
                ],
                dim=0,
            ),
        ),
        (
            "env_three_cubes_grids",
            [
                [[-1, 0, [0, 0, 0], [1, -1, -1]], {0: [0, 0]}],
                [[-1, 0, [0, 0, 0], [1, 0, -1]], {0: [0, 0], 1: [-1, -1]}],
            ],
            torch.stack(
                [
                    torch.tensor(
                        [
                            # fmt: off
                            0.0, 0.0, 0.0, # ACTIVE SUBENV
                            0.0, # TOGGLE FLAG
                            0.0, 0.0, 0.0, # DONES
                            0.0, 1.0, # PRESENT SUBENVS OF EACH TYPE
                            # CUBE 0
                            -1.0, -1.0,
                            # CUBE 1
                            -1.0, -1.0,
                            # CUBE 2
                            -1.0, -1.0,
                            # GRID 0
                            1.0, 0.0, 0.0,
                            1.0, 0.0, 0.0,
                            # GRID 1
                            1.0, 0.0, 0.0,
                            1.0, 0.0, 0.0,
                            # GRID 2
                            1.0, 0.0, 0.0,
                            1.0, 0.0, 0.0,
                            # fmt: on
                        ],
                        dtype=torch.float,
                    ),
                    torch.tensor(
                        [
                            # fmt: off
                            0.0, 0.0, 0.0, # ACTIVE SUBENV
                            0.0, # TOGGLE FLAG
                            0.0, 0.0, 0.0, # DONES
                            1.0, 1.0, # PRESENT SUBENVS OF EACH TYPE
                            # CUBE 0
                            -1.0, -1.0,
                            # CUBE 1
                            -1.0, -1.0,
                            # CUBE 2
                            -1.0, -1.0,
                            # GRID 0
                            1.0, 0.0, 0.0,
                            1.0, 0.0, 0.0,
                            # GRID 1
                            1.0, 0.0, 0.0,
                            1.0, 0.0, 0.0,
                            # GRID 2
                            1.0, 0.0, 0.0,
                            1.0, 0.0, 0.0,
                            # fmt: on
                        ],
                        dtype=torch.float,
                    ),
                ],
                dim=0,
            ),
        ),
        (
            "env_three_cubes_grids",
            [
                [[-1, 0, [0, 0, 0], [1, -1, -1]], {0: [0, 0]}],
                [[-1, 0, [0, 0, 0], [1, 0, -1]], {0: [0, 0], 1: [-1, -1]}],
                [[1, 1, [1, 0, 0], [1, 0, 1]], {0: [0, 1], 1: [0.2, 0.6], 2: [1, 2]}],
            ],
            torch.stack(
                [
                    torch.tensor(
                        [
                            # fmt: off
                            0.0, 0.0, 0.0, # ACTIVE SUBENV
                            0.0, # TOGGLE FLAG
                            0.0, 0.0, 0.0, # DONES
                            0.0, 1.0, # PRESENT SUBENVS OF EACH TYPE
                            # CUBE 0
                            -1.0, -1.0,
                            # CUBE 1
                            -1.0, -1.0,
                            # CUBE 2
                            -1.0, -1.0,
                            # GRID 0
                            1.0, 0.0, 0.0,
                            1.0, 0.0, 0.0,
                            # GRID 1
                            1.0, 0.0, 0.0,
                            1.0, 0.0, 0.0,
                            # GRID 2
                            1.0, 0.0, 0.0,
                            1.0, 0.0, 0.0,
                            # fmt: on
                        ],
                        dtype=torch.float,
                    ),
                    torch.tensor(
                        [
                            # fmt: off
                            0.0, 0.0, 0.0, # ACTIVE SUBENV
                            0.0, # TOGGLE FLAG
                            0.0, 0.0, 0.0, # DONES
                            1.0, 1.0, # PRESENT SUBENVS OF EACH TYPE
                            # CUBE 0
                            -1.0, -1.0,
                            # CUBE 1
                            -1.0, -1.0,
                            # CUBE 2
                            -1.0, -1.0,
                            # GRID 0
                            1.0, 0.0, 0.0,
                            1.0, 0.0, 0.0,
                            # GRID 1
                            1.0, 0.0, 0.0,
                            1.0, 0.0, 0.0,
                            # GRID 2
                            1.0, 0.0, 0.0,
                            1.0, 0.0, 0.0,
                            # fmt: on
                        ],
                        dtype=torch.float,
                    ),
                    torch.tensor(
                        [
                            # fmt: off
                            0.0, 1.0, 0.0, # ACTIVE SUBENV
                            1.0, # TOGGLE FLAG
                            1.0, 0.0, 0.0, # DONES
                            1.0, 2.0, # PRESENT SUBENVS OF EACH TYPE
                            # CUBE 0
                            -0.6, 0.2,
                            # CUBE 1
                            -1.0, -1.0,
                            # CUBE 2
                            -1.0, -1.0,
                            # GRID 0
                            1.0, 0.0, 0.0,
                            0.0, 1.0, 0.0,
                            # GRID 1
                            0.0, 1.0, 0.0,
                            0.0, 0.0, 1.0,
                            # GRID 2
                            1.0, 0.0, 0.0,
                            1.0, 0.0, 0.0,
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
    states_policy = env.states2policy(states)
    assert torch.all(torch.isclose(states_policy_exp, env.states2policy(states)))


class TestSetFlexTwoGrids(common.BaseTestsDiscrete):
    """Common tests for setflex of up to two grids."""

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
            "test__gflownet_minimal_runs": 0,
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


class TestSetFlexThreeCubes(common.BaseTestsContinuous):
    """Common tests for setflex of up to three cubes."""

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
            "test__gflownet_minimal_runs": 0,
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


class TestSetFlexThreeCubesGrids(common.BaseTestsContinuous):
    """Common tests for setflex of up to three cubes and grids."""

    @pytest.fixture(autouse=True)
    def setup(self, env_three_cubes_grids):
        self.env = env_three_cubes_grids
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
            "test__gflownet_minimal_runs": 0,
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


class TestSetFlexThreeStacksDiff(common.BaseTestsContinuous):
    """Common tests for setflex of up to three different stacks."""

    @pytest.fixture(autouse=True)
    def setup(self, env_three_stacks_diff):
        self.env = env_three_stacks_diff
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
            "test__gflownet_minimal_runs": 0,
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


class TestSetFlexThreeOrbitLikeStacks(common.BaseTestsContinuous):
    """Common tests for setflex with three orbit-like stacks."""

    @pytest.fixture(autouse=True)
    def setup(self, env_three_orbit_like_stacks):
        self.env = env_three_orbit_like_stacks
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
            "test__gflownet_minimal_runs": 0,
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
