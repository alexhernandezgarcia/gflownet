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
            # fmt: off
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
        ]
    ],
)
def test__get_action_space__returns_expected(env_grid2d_tetrismini, action_space):
    env = env_grid2d_tetrismini
    assert set(action_space) == set(env.action_space)


@pytest.mark.parametrize(
    "env", ["env_grid2d_tetrismini", "env_cube_tetris", "env_cube_tetris_grid"]
)
def test__action_space__contains_actions_of_all_subenvs(env, request):
    env = request.getfixturevalue(env)
    action_space = env.action_space
    for stage, subenv in env.subenvs.items():
        assert all(
            [
                env._pad_action(action, stage) in action_space
                for action in subenv.action_space
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
            [
                # fmt: off
                0,
                [0, 0],
                torch.tensor([
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                ], dtype=torch.int16, device="cpu"),
                # fmt: on
            ],
        ),
        (
            "env_cube_tetris",
            [
                # fmt: off
                0,
                [-1.0, -1.0],
                torch.tensor([
                    [000, 000],
                    [000, 000],
                    [000, 000],
                    [000, 000],
                    [000, 000],
                    [000, 000],
                ], dtype=torch.int16, device="cpu"),
                # fmt: on
            ],
        ),
        (
            "env_cube_tetris_grid",
            [
                # fmt: off
                0,
                [-1.0, -1.0],
                torch.tensor([
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                ], dtype=torch.int16, device="cpu"),
                [0, 0, 0],
                # fmt: on
            ],
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
            [
                # fmt: off
                0,
                [0, 0],
                torch.tensor([
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                ], dtype=torch.int16, device="cpu"),
                # fmt: on
            ],
            True,
        ),
        (
            "env_cube_tetris",
            [
                # fmt: off
                0,
                [-1.0, -1.0],
                torch.tensor([
                    [000, 000],
                    [000, 000],
                    [000, 000],
                    [000, 000],
                    [000, 000],
                    [000, 000],
                ], dtype=torch.int16, device="cpu"),
                # fmt: on
            ],
            True,
        ),
        (
            "env_cube_tetris_grid",
            [
                # fmt: off
                0,
                [-1.0, -1.0],
                torch.tensor([
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                ], dtype=torch.int16, device="cpu"),
                [0, 0, 0],
                # fmt: on
            ],
            True,
        ),
        (
            "env_cube_tetris_grid",
            [
                # fmt: off
                1,
                [-1.0, -1.0],
                torch.tensor([
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                ], dtype=torch.int16, device="cpu"),
                [0, 0, 0],
                # fmt: on
            ],
            False,
        ),
        (
            "env_cube_setgrids",
            [
                # fmt: off
                0,
                [-1, -1],
                [[-1, 0, [0, 0], [0, 0]], {0: [0, 0], 1: [0, 0]}],
                # fmt: on
            ],
            True,
        ),
        (
            "env_cube_setgrids",
            [
                # fmt: off
                1,
                [-1, -1],
                [[-1, 0, [0, 0], [0, 0]], {0: [0, 0], 1: [0, 0]}],
                # fmt: on
            ],
            False,
        ),
        (
            "env_cube_setgrids",
            [
                # fmt: off
                0,
                [0.0, 0.0],
                [[-1, 0, [0, 0], [0, 0]], {0: [0, 0], 1: [0, 0]}],
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
    "env, state, dones",
    [
        (
            "env_grid2d_tetrismini",
            [
                # fmt: off
                0,
                [0, 0],
                torch.tensor([
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                ], dtype=torch.int16, device="cpu"),
                # fmt: on
            ],
            [False, False],
        ),
        (
            "env_grid2d_tetrismini",
            [
                # fmt: off
                1,
                [0, 0],
                torch.tensor([
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                ], dtype=torch.int16, device="cpu"),
                # fmt: on
            ],
            [True, False],
        ),
        (
            "env_grid2d_tetrismini",
            [
                # fmt: off
                0,
                [1, 2],
                torch.tensor([
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                ], dtype=torch.int16, device="cpu"),
                # fmt: on
            ],
            [False, False],
        ),
        (
            "env_grid2d_tetrismini",
            [
                # fmt: off
                1,
                [1, 2],
                torch.tensor([
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                ], dtype=torch.int16, device="cpu"),
                # fmt: on
            ],
            [True, False],
        ),
        (
            "env_grid2d_tetrismini",
            [
                # fmt: off
                1,
                [1, 2],
                torch.tensor([
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 400, 400, 000],
                    [000, 400, 400, 000],
                ], dtype=torch.int16, device="cpu"),
                # fmt: on
            ],
            [True, False],
        ),
        (
            "env_grid2d_tetrismini",
            [
                # fmt: off
                1,
                [1, 2],
                torch.tensor([
                    [000, 000, 000, 000],
                    [100, 000, 000, 000],
                    [100, 000, 000, 000],
                    [100, 400, 400, 000],
                    [100, 400, 400, 000],
                ], dtype=torch.int16, device="cpu"),
                # fmt: on
            ],
            [True, False],
        ),
        (
            "env_grid2d_tetrismini",
            [
                # fmt: off
                1,
                [1, 2],
                torch.tensor([
                    [000, 000, 000, 000],
                    [100, 000, 401, 401],
                    [100, 000, 401, 401],
                    [100, 400, 400, 000],
                    [100, 400, 400, 000],
                ], dtype=torch.int16, device="cpu"),
                # fmt: on
            ],
            [True, True],
        ),
        (
            "env_cube_setgrids",
            [
                # fmt: off
                0,
                [-1, -1],
                [[-1, 0, [0, 0], [0, 0]], {0: [0, 0], 1: [0, 0]}],
                # fmt: on
            ],
            [False, False],
        ),
        (
            "env_cube_setgrids",
            [
                # fmt: off
                0,
                [0.3, 0.7],
                [[-1, 0, [0, 0], [0, 0]], {0: [0, 0], 1: [0, 0]}],
                # fmt: on
            ],
            [False, False],
        ),
        (
            "env_cube_setgrids",
            [
                # fmt: off
                1,
                [0.3, 0.7],
                [[-1, 0, [0, 0], [0, 0]], {0: [0, 0], 1: [0, 0]}],
                # fmt: on
            ],
            [True, False],
        ),
        (
            "env_cube_setgrids",
            [
                # fmt: off
                1,
                [0.3, 0.7],
                [[0, 0, [0, 0], [0, 0]], {0: [1, 0], 1: [0, 0]}],
                # fmt: on
            ],
            [True, False],
        ),
        (
            "env_cube_setgrids",
            [
                # fmt: off
                1,
                [0.3, 0.7],
                [[-1, 0, [1, 1], [0, 0]], {0: [1, 2], 1: [1, 1]}],
                # fmt: on
            ],
            [True, True],
        ),
        (
            "env_cube_setstacks",
            [
                0,
                [-1, -1],
                [
                    # fmt: off
                    [-1, 0, [0, 0], [0, 1]],
                    {
                        0: [0, [-1.0, -1.0], [0, 0]],
                        1: [0, [0, 0], [-1.0, -1.0]]
                    },
                    # fmt: on
                ],
            ],
            [False, False],
        ),
        (
            "env_cube_setstacks",
            [
                0,
                [0.3, 0.7],
                [
                    # fmt: off
                    [-1, 0, [0, 0], [0, 1]],
                    {
                        0: [0, [-1.0, -1.0], [0, 0]],
                        1: [0, [0, 0], [-1.0, -1.0]]
                    },
                    # fmt: on
                ],
            ],
            [False, False],
        ),
        (
            "env_cube_setstacks",
            [
                1,
                [0.3, 0.7],
                [
                    # fmt: off
                    [-1, 0, [0, 0], [0, 1]],
                    {
                        0: [0, [-1.0, -1.0], [0, 0]],
                        1: [0, [0, 0], [-1.0, -1.0]]
                    },
                    # fmt: on
                ],
            ],
            [True, False],
        ),
        (
            "env_cube_setstacks",
            [
                1,
                [0.3, 0.7],
                [
                    # fmt: off
                    [0, 0, [0, 0], [0, 1]],
                    {
                        0: [0, [0.1, 0.2], [0, 0]],
                        1: [0, [0, 0], [-1.0, -1.0]]
                    },
                    # fmt: on
                ],
            ],
            [True, False],
        ),
        (
            "env_cube_setstacks",
            [
                1,
                [0.3, 0.7],
                [
                    # fmt: off
                    [-1, 0, [1, 0], [0, 1]],
                    {
                        0: [1, [0.1, 0.2], [2, 1]],
                        1: [0, [0, 0], [-1.0, -1.0]]
                    },
                    # fmt: on
                ],
            ],
            [True, False],
        ),
        (
            "env_cube_setstacks",
            [
                1,
                [0.3, 0.7],
                [
                    # fmt: off
                    [-1, 0, [1, 1], [0, 1]],
                    {
                        0: [1, [0.1, 0.2], [2, 1]],
                        1: [0, [1, 2], [0.3, 0.8]]
                    },
                    # fmt: on
                ],
            ],
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
    for stage, subenv in env.subenvs.items():
        assert env.equal(subenv.state, env._get_substate(state, stage))

    # Check dones
    for subenv, done in zip(env.subenvs.values(), dones):
        assert subenv.done == done


@pytest.mark.skip(reason="skip while developping other tests")
@pytest.mark.parametrize(
    "env, state",
    [
        (
            "env_grid2d_tetrismini",
            [
                # fmt: off
                0,
                [2, 0],
                torch.tensor([
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                ], dtype=torch.int16, device="cpu"),
                # fmt: on
            ],
        ),
        (
            "env_grid2d_tetrismini",
            [
                # fmt: off
                1,
                [2, 0],
                torch.tensor([
                    [000, 000, 000, 000],
                    [100, 000, 000, 000],
                    [100, 000, 000, 000],
                    [100, 000, 000, 000],
                    [100, 000, 000, 000],
                ], dtype=torch.int16, device="cpu"),
                # fmt: on
            ],
        ),
        (
            "env_cube_tetris",
            [
                # fmt: off
                0,
                [0.3, 0.7],
                torch.tensor([
                    [000, 000],
                    [000, 000],
                    [000, 000],
                    [000, 000],
                    [000, 000],
                    [000, 000],
                ], dtype=torch.int16, device="cpu"),
                # fmt: on
            ],
        ),
        (
            "env_cube_tetris",
            [
                # fmt: off
                1,
                [0.3, 0.7],
                torch.tensor([
                    [000, 200],
                    [000, 200],
                    [200, 200],
                    [300, 000],
                    [300, 000],
                    [300, 300],
                ], dtype=torch.int16, device="cpu"),
                # fmt: on
            ],
        ),
        (
            "env_cube_tetris_grid",
            [
                # fmt: off
                2,
                [0.3, 0.7],
                torch.tensor([
                    [000, 200],
                    [000, 200],
                    [200, 200],
                    [300, 000],
                    [300, 000],
                    [300, 300],
                ], dtype=torch.int16, device="cpu"),
                [1, 0, 2],
                # fmt: on
            ],
        ),
    ],
)
def test__get_mask_invalid_actions_backward__returns_expected_general_case(
    env, state, request
):
    env = request.getfixturevalue(env)
    stage = env._get_stage(state)
    mask = env.get_mask_invalid_actions_backward(state, done=False)
    for stg, subenv in env.subenvs.items():
        if stg == stage:
            # Mask of state if stage is current stage in state
            mask_subenv_expected = subenv.get_mask_invalid_actions_backward(
                env._get_substate(state, stg)
            )
        else:
            # Dummy mask (all True) if stage is other than current stage in state
            mask_subenv_expected = [True] * subenv.mask_dim
        mask_subenv = env._get_mask_of_subenv(mask, stg)
        assert mask_subenv == mask_subenv_expected, state


@pytest.mark.parametrize(
    "env, state, dones",
    [
        # Tetris source, Tetris not done
        (
            "env_grid2d_tetrismini",
            [
                # fmt: off
                1,
                [2, 0],
                torch.tensor([
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                ], dtype=torch.int16, device="cpu"),
                # fmt: on
            ],
            [True, False],
        ),
        # Tetris source, Tetris done (only valid action is Tetris EOS)
        (
            "env_grid2d_tetrismini",
            [
                # fmt: off
                1,
                [2, 0],
                torch.tensor([
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                ], dtype=torch.int16, device="cpu"),
                # fmt: on
            ],
            [True, True],
        ),
        # Global source
        (
            "env_grid2d_tetrismini",
            [
                # fmt: off
                0,
                [0, 0],
                torch.tensor([
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                ], dtype=torch.int16, device="cpu"),
                # fmt: on
            ],
            [False, False],
        ),
        (
            "env_cube_tetris",
            [
                # fmt: off
                1,
                [0.3, 0.7],
                torch.tensor([
                    [000, 000],
                    [000, 000],
                    [000, 000],
                    [000, 000],
                    [000, 000],
                    [000, 000],
                ], dtype=torch.int16, device="cpu"),
                # fmt: on
            ],
            [True, False],
        ),
        # Last stage is source but done
        (
            "env_cube_tetris_grid",
            [
                # fmt: off
                2,
                [0.3, 0.7],
                torch.tensor([
                    [000, 200],
                    [000, 200],
                    [200, 200],
                    [300, 000],
                    [300, 000],
                    [300, 300],
                ], dtype=torch.int16, device="cpu"),
                [0, 0, 0],
                # fmt: on
            ],
            [True, True, True],
        ),
        # Last stage is source, but not done
        (
            "env_cube_tetris_grid",
            [
                # fmt: off
                2,
                [0.3, 0.7],
                torch.tensor([
                    [000, 200],
                    [000, 200],
                    [200, 200],
                    [300, 000],
                    [300, 000],
                    [300, 300],
                ], dtype=torch.int16, device="cpu"),
                [0, 0, 0],
                # fmt: on
            ],
            [True, True, False],
        ),
    ],
)
def test__get_mask_invalid_actions_backward__returns_expected_stage_transition(
    env, state, dones, request
):
    env = request.getfixturevalue(env)
    stage = env._get_stage(state)
    subenv = env.subenvs[stage]
    state_subenv = env._get_substate(state, stage)
    done = dones[-1]
    # If it is not the initial stage, the env is not done and the state of the subenv
    # is the source, then the relevant mask is the one from the previous subenv
    if stage > 0 and not done and subenv.equal(state_subenv, subenv.source):
        stage -= 1
        subenv = env.subenvs[stage]
        state_subenv = env._get_substate(state, stage)
        done = True
    # Get the global mask and extract the relenvant part
    mask = env.get_mask_invalid_actions_backward(state, done=dones[-1])
    mask_subenv = mask[env.n_subenvs : env.n_subenvs + subenv.mask_dim]
    # Get expected mask of the subenv
    mask_subenv_expected = subenv.get_mask_invalid_actions_backward(
        env._get_substate(state, stage), done=done
    )
    assert mask_subenv == mask_subenv_expected, state


@pytest.mark.parametrize(
    "env, state_from, action, state_next_exp, valid_exp",
    [
        (
            "env_grid2d_tetrismini",
            [
                # fmt: off
                0,
                [0, 0],
                torch.tensor([
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                ], dtype=torch.int16, device="cpu"),
                # fmt: on
            ],
            (0, 1, 0, 0),
            [
                # fmt: off
                0,
                [1, 0],
                torch.tensor([
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                ], dtype=torch.int16, device="cpu"),
                # fmt: on
            ],
            True,
        ),
        (
            "env_grid2d_tetrismini",
            [
                # fmt: off
                0,
                [1, 0],
                torch.tensor([
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                ], dtype=torch.int16, device="cpu"),
                # fmt: on
            ],
            (0, 1, 0, 0),
            [
                # fmt: off
                0,
                [2, 0],
                torch.tensor([
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                ], dtype=torch.int16, device="cpu"),
                # fmt: on
            ],
            True,
        ),
        # EOS action from Grid must increment stage
        (
            "env_grid2d_tetrismini",
            [
                # fmt: off
                0,
                [2, 0],
                torch.tensor([
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                ], dtype=torch.int16, device="cpu"),
                # fmt: on
            ],
            (0, 0, 0, 0),
            [
                # fmt: off
                1,
                [2, 0],
                torch.tensor([
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                ], dtype=torch.int16, device="cpu"),
                # fmt: on
            ],
            True,
        ),
        (
            "env_grid2d_tetrismini",
            [
                # fmt: off
                1,
                [2, 0],
                torch.tensor([
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                ], dtype=torch.int16, device="cpu"),
                # fmt: on
            ],
            (1, 1, 0, 0),
            [
                # fmt: off
                1,
                [2, 0],
                torch.tensor([
                    [000, 000, 000, 000],
                    [100, 000, 000, 000],
                    [100, 000, 000, 000],
                    [100, 000, 000, 000],
                    [100, 000, 000, 000],
                ], dtype=torch.int16, device="cpu"),
                # fmt: on
            ],
            True,
        ),
        (
            "env_grid2d_tetrismini",
            [
                # fmt: off
                1,
                [2, 0],
                torch.tensor([
                    [000, 000, 000, 000],
                    [100, 000, 000, 000],
                    [100, 000, 000, 000],
                    [100, 000, 000, 000],
                    [100, 000, 000, 000],
                ], dtype=torch.int16, device="cpu"),
                # fmt: on
            ],
            (1, 4, 0, 2),
            [
                # fmt: off
                1,
                [2, 0],
                torch.tensor([
                    [000, 000, 000, 000],
                    [100, 000, 000, 000],
                    [100, 000, 000, 000],
                    [100, 000, 400, 400],
                    [100, 000, 400, 400],
                ], dtype=torch.int16, device="cpu"),
                # fmt: on
            ],
            True,
        ),
        # EOS from last stage
        (
            "env_grid2d_tetrismini",
            [
                # fmt: off
                1,
                [2, 0],
                torch.tensor([
                    [000, 000, 000, 000],
                    [100, 401, 401, 000],
                    [100, 401, 401, 000],
                    [100, 000, 400, 400],
                    [100, 000, 400, 400],
                ], dtype=torch.int16, device="cpu"),
                # fmt: on
            ],
            (1, -1, -1, -1),
            [
                # fmt: off
                1,
                [2, 0],
                torch.tensor([
                    [000, 000, 000, 000],
                    [100, 401, 401, 000],
                    [100, 401, 401, 000],
                    [100, 000, 400, 400],
                    [100, 000, 400, 400],
                ], dtype=torch.int16, device="cpu"),
                # fmt: on
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
            [
                # fmt: off
                0,
                [2, 0],
                torch.tensor([
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                ], dtype=torch.int16, device="cpu"),
                # fmt: on
            ],
            (0, 1, 0, 0),
            [
                # fmt: off
                0,
                [1, 0],
                torch.tensor([
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                ], dtype=torch.int16, device="cpu"),
                # fmt: on
            ],
            True,
        ),
        (
            "env_grid2d_tetrismini",
            [
                # fmt: off
                1,
                [2, 0],
                torch.tensor([
                    [000, 000, 000, 000],
                    [100, 000, 000, 000],
                    [100, 000, 000, 000],
                    [100, 000, 000, 000],
                    [100, 000, 000, 000],
                ], dtype=torch.int16, device="cpu"),
                # fmt: on
            ],
            (1, 1, 0, 0),
            [
                # fmt: off
                1,
                [2, 0],
                torch.tensor([
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                ], dtype=torch.int16, device="cpu"),
                # fmt: on
            ],
            True,
        ),
        (
            "env_grid2d_tetrismini",
            [
                # fmt: off
                1,
                [2, 0],
                torch.tensor([
                    [000, 000, 000, 000],
                    [100, 000, 000, 000],
                    [100, 000, 000, 000],
                    [100, 000, 000, 000],
                    [100, 000, 000, 000],
                ], dtype=torch.int16, device="cpu"),
                # fmt: on
            ],
            (1, 4, 0, 2),
            [
                # fmt: off
                1,
                [2, 0],
                torch.tensor([
                    [000, 000, 000, 000],
                    [100, 000, 000, 000],
                    [100, 000, 000, 000],
                    [100, 000, 000, 000],
                    [100, 000, 000, 000],
                ], dtype=torch.int16, device="cpu"),
                # fmt: on
            ],
            False,
        ),
        # Back to source
        (
            "env_grid2d_tetrismini",
            [
                # fmt: off
                0,
                [1, 0],
                torch.tensor([
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                ], dtype=torch.int16, device="cpu"),
                # fmt: on
            ],
            (0, 1, 0, 0),
            [
                # fmt: off
                0,
                [0, 0],
                torch.tensor([
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                ], dtype=torch.int16, device="cpu"),
                # fmt: on
            ],
            True,
        ),
        # Action from source must be invalid
        (
            "env_grid2d_tetrismini",
            [
                # fmt: off
                0,
                [0, 0],
                torch.tensor([
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                ], dtype=torch.int16, device="cpu"),
                # fmt: on
            ],
            (0, 1, 0, 0),
            [
                # fmt: off
                0,
                [0, 0],
                torch.tensor([
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                ], dtype=torch.int16, device="cpu"),
                # fmt: on
            ],
            False,
        ),
        # Grid EOS from Tetris
        (
            "env_grid2d_tetrismini",
            [
                # fmt: off
                1,
                [2, 0],
                torch.tensor([
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                ], dtype=torch.int16, device="cpu"),
                # fmt: on
            ],
            (0, 0, 0, 0),
            [
                # fmt: off
                0,
                [2, 0],
                torch.tensor([
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                ], dtype=torch.int16, device="cpu"),
                # fmt: on
            ],
            True,
        ),
        # Global EOS
        (
            "env_grid2d_tetrismini",
            [
                # fmt: off
                1,
                [2, 0],
                torch.tensor([
                    [000, 000, 000, 000],
                    [100, 401, 401, 000],
                    [100, 401, 401, 000],
                    [100, 000, 400, 400],
                    [100, 000, 400, 400],
                ], dtype=torch.int16, device="cpu"),
                # fmt: on
            ],
            (1, -1, -1, -1),
            [
                # fmt: off
                1,
                [2, 0],
                torch.tensor([
                    [000, 000, 000, 000],
                    [100, 401, 401, 000],
                    [100, 401, 401, 000],
                    [100, 000, 400, 400],
                    [100, 000, 400, 400],
                ], dtype=torch.int16, device="cpu"),
                # fmt: on
            ],
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
            [
                # fmt: off
                0,
                [0, 0],
                torch.tensor([
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                ], dtype=torch.int16, device="cpu"),
                # fmt: on
            ],
            [],
            [],
        ),
        # Intermediate grid
        (
            "env_grid2d_tetrismini",
            [
                # fmt: off
                0,
                [1, 2],
                torch.tensor([
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                ], dtype=torch.int16, device="cpu"),
                # fmt: on
            ],
            [
                [
                    # fmt: off
                    0,
                    [0, 2],
                    torch.tensor([
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                    ], dtype=torch.int16, device="cpu"),
                    # fmt: on
                ],
                [
                    # fmt: off
                    0,
                    [1, 1],
                    torch.tensor([
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                    ], dtype=torch.int16, device="cpu"),
                    # fmt: on
                ],
            ],
            [(0, 1, 0, 0), (0, 0, 1, 0)],
        ),
        # Source Tetris
        (
            "env_grid2d_tetrismini",
            [
                # fmt: off
                1,
                [1, 2],
                torch.tensor([
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                ], dtype=torch.int16, device="cpu"),
                # fmt: on
            ],
            [
                [
                    # fmt: off
                    0,
                    [1, 2],
                    torch.tensor([
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                        [000, 000, 000, 000],
                    ], dtype=torch.int16, device="cpu"),
                    # fmt: on
                ],
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
        parents_actions_exp_dict[action] = tuple(parent.copy())
    parents_actions_dict = {}
    for parent, action in zip(parents, parent_actions):
        parents_actions_dict[action] = tuple(parent.copy())

    assert sorted(parents_actions_exp_dict) == sorted(parents_actions_dict)


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
