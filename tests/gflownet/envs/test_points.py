import common
import numpy as np
import pytest
import torch

from gflownet.envs.points import Points
from gflownet.utils.common import tbool, tfloat


@pytest.fixture
def cont1p1d():
    return Points(
        n_points=1,
        n_dim=1,
        cube_mode="continuous",
        cube_kwargs={"n_comp": 3, "min_incr": 0.1},
    )


@pytest.fixture
def hybrid1p1d():
    return Points(
        n_points=1,
        n_dim=1,
        cube_mode="hybrid",
        cube_kwargs={"n_comp": 3, "min_incr": 0.1},
    )


@pytest.fixture
def cont1p2d():
    return Points(
        n_points=1,
        n_dim=2,
        cube_mode="continuous",
        cube_kwargs={"n_comp": 3, "min_incr": 0.1},
    )


@pytest.fixture
def hybrid1p2d():
    return Points(
        n_points=1,
        n_dim=2,
        cube_mode="hybrid",
        cube_kwargs={"n_comp": 3, "min_incr": 0.1},
    )


@pytest.fixture
def cont2p2d():
    return Points(
        n_points=2,
        n_dim=2,
        cube_mode="continuous",
        cube_kwargs={"n_comp": 3, "min_incr": 0.1},
    )


@pytest.fixture
def hybrid2p2d():
    return Points(
        n_points=2,
        n_dim=2,
        cube_mode="hybrid",
        cube_kwargs={"n_comp": 3, "min_incr": 0.1},
    )


@pytest.fixture
def cont7p5d():
    return Points(
        n_points=7,
        n_dim=5,
        cube_mode="continuous",
        cube_kwargs={"n_comp": 3, "min_incr": 0.1},
    )


@pytest.mark.parametrize(
    "env",
    [
        "cont1p1d",
        "hybrid1p1d",
        "cont1p2d",
        "hybrid1p2d",
        "cont2p2d",
        "hybrid2p2d",
        "cont7p5d",
    ],
)
def test__environment__initializes_properly(env, request):
    env = request.getfixturevalue(env)
    assert True


@pytest.mark.parametrize(
    "env, action_space",
    [
        (
            "cont1p1d",
            [
                (-1, 0, 0),
                (-1, -1, -1),
                (0, 0.0, 0),
                (0, 0.0, 1),
                (0, np.inf, np.inf),
            ],
        ),
        (
            "cont7p5d",
            [
                (-1, 0, 0, 0, 0, 0, 0),
                (-1, 1, 0, 0, 0, 0, 0),
                (-1, 2, 0, 0, 0, 0, 0),
                (-1, 3, 0, 0, 0, 0, 0),
                (-1, 4, 0, 0, 0, 0, 0),
                (-1, 5, 0, 0, 0, 0, 0),
                (-1, 6, 0, 0, 0, 0, 0),
                (-1, -1, -1, -1, -1, -1, -1),
                (0, 0.0, 0.0, 0.0, 0.0, 0.0, 0),
                (0, 0.0, 0.0, 0.0, 0.0, 0.0, 1),
                (0, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf),
            ],
        ),
    ],
)
def test__get_action_space__returns_expected(env, action_space, request):
    env = request.getfixturevalue(env)
    assert action_space == env.action_space
