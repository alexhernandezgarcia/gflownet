"""
These tests are for the continuous lattice parameters environment. The tests for the
former discrete implementation have been removed for simplicity.  Check commit
9f3477d8e46c4624f9162d755663993b83196546 to see these changes or the history previous
to that commit to consult previous implementations.
"""

import common
import pytest
import torch

from gflownet.envs.crystals.lattice_parameters import (
    LATTICE_SYSTEM_INDEX,
    PARAMETER_NAMES,
    LatticeParameters,
)
from gflownet.utils.common import tfloat
from gflownet.utils.crystals.constants import (
    CUBIC,
    HEXAGONAL,
    LATTICE_SYSTEMS,
    MONOCLINIC,
    ORTHORHOMBIC,
    RHOMBOHEDRAL,
    TETRAGONAL,
    TRICLINIC,
)

N_REPETITIONS = 100


@pytest.fixture()
def env(lattice_system):
    return LatticeParameters(
        lattice_system=lattice_system,
        min_length=1.0,
        max_length=5.0,
        min_angle=30.0,
        max_angle=150.0,
    )


@pytest.mark.parametrize("lattice_system", LATTICE_SYSTEMS)
def test__environment__initializes_properly(env, lattice_system):
    assert True


@pytest.mark.parametrize(
    "lattice_system",
    [CUBIC],
)
@pytest.mark.repeat(N_REPETITIONS)
def test__cubic__constraints_remain_after_random_trajectory(env, lattice_system):
    env = env.reset()
    while not env.done:
        env.trajectory_random()
        (a, b, c), (alpha, beta, gamma) = env._get_lengths_angles()
        assert len({a, b, c}) == 1
        assert len({alpha, beta, gamma, 90.0}) == 1


@pytest.mark.parametrize(
    "lattice_system",
    [HEXAGONAL],
)
@pytest.mark.repeat(N_REPETITIONS)
def test__hexagonal__constraints_remain_after_random_trajectory(env, lattice_system):
    env = env.reset()
    while not env.done:
        env.trajectory_random()
        (a, b, c), (alpha, beta, gamma) = env._get_lengths_angles()
        assert a == b
        assert len({a, b, c}) == 2
        assert len({alpha, beta, 90.0}) == 1
        assert gamma == 120.0


@pytest.mark.parametrize(
    "lattice_system",
    [MONOCLINIC],
)
@pytest.mark.repeat(N_REPETITIONS)
def test__monoclinic__constraints_remain_after_random_trajectory(env, lattice_system):
    env = env.reset()
    while not env.done:
        env.trajectory_random()
        (a, b, c), (alpha, beta, gamma) = env._get_lengths_angles()
        assert len({a, b, c}) == 3
        assert len({alpha, gamma, 90.0}) == 1
        assert beta != 90.0


@pytest.mark.parametrize(
    "lattice_system",
    [ORTHORHOMBIC],
)
@pytest.mark.repeat(N_REPETITIONS)
def test__orthorhombic__constraints_remain_after_random_trajectory(env, lattice_system):
    env = env.reset()
    while not env.done:
        env.trajectory_random()
        (a, b, c), (alpha, beta, gamma) = env._get_lengths_angles()
        assert len({a, b, c}) == 3
        assert len({alpha, beta, gamma, 90.0}) == 1


@pytest.mark.parametrize(
    "lattice_system",
    [RHOMBOHEDRAL],
)
@pytest.mark.repeat(N_REPETITIONS)
def test__rhombohedral__constraints_remain_after_random_trajectory(env, lattice_system):
    env = env.reset()
    while not env.done:
        env.trajectory_random()
        (a, b, c), (alpha, beta, gamma) = env._get_lengths_angles()
        assert len({a, b, c}) == 1
        assert len({alpha, beta, gamma}) == 1
        assert len({alpha, beta, gamma, 90.0}) == 2


@pytest.mark.parametrize(
    "lattice_system",
    [TETRAGONAL],
)
@pytest.mark.repeat(N_REPETITIONS)
def test__tetragonal__constraints_remain_after_random_trajectory(env, lattice_system):
    env = env.reset()
    while not env.done:
        env.trajectory_random()
        (a, b, c), (alpha, beta, gamma) = env._get_lengths_angles()
        assert a == b
        assert len({a, b, c}) == 2
        assert len({alpha, beta, gamma, 90.0}) == 1


@pytest.mark.parametrize(
    "lattice_system",
    [TRICLINIC],
)
@pytest.mark.repeat(N_REPETITIONS)
def test__triclinic__constraints_remain_after_random_trajectory(env, lattice_system):
    env = env.reset()
    while not env.done:
        env.trajectory_random()
        (a, b, c), (alpha, beta, gamma) = env._get_lengths_angles()
        assert len({a, b, c}) == 3
        assert len({alpha, beta, gamma, 90.0}) == 4


@pytest.mark.parametrize(
    "lattice_system, states, expected",
    [
        (
            TRICLINIC,
            [
                [1, [LATTICE_SYSTEM_INDEX[TRICLINIC]], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                [1, [LATTICE_SYSTEM_INDEX[TRICLINIC]], [0.0, 0.2, 0.5, 0.0, 0.5, 1.0]],
                [1, [LATTICE_SYSTEM_INDEX[TRICLINIC]], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
                [0, [LATTICE_SYSTEM_INDEX[TRICLINIC]], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                [0, [LATTICE_SYSTEM_INDEX[TRICLINIC]], [0.0, 0.2, 0.5, 0.0, 0.5, 1.0]],
                [0, [LATTICE_SYSTEM_INDEX[TRICLINIC]], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
            ],
            [
                [1.0, 1.0, 1.0, 30.0, 30.0, 30.0],
                [1.0, 1.8, 3.0, 30.0, 90.0, 150.0],
                [5.0, 5.0, 5.0, 150.0, 150.0, 150.0],
                [1.0, 1.0, 1.0, 30.0, 30.0, 30.0],
                [1.0, 1.8, 3.0, 30.0, 90.0, 150.0],
                [5.0, 5.0, 5.0, 150.0, 150.0, 150.0],
            ],
        ),
        (
            CUBIC,
            [
                [1, [LATTICE_SYSTEM_INDEX[TRICLINIC]], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                [1, [LATTICE_SYSTEM_INDEX[TRICLINIC]], [0.0, 0.2, 0.5, 0.0, 0.5, 1.0]],
                [1, [LATTICE_SYSTEM_INDEX[TRICLINIC]], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
                [0, [LATTICE_SYSTEM_INDEX[TRICLINIC]], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                [0, [LATTICE_SYSTEM_INDEX[TRICLINIC]], [0.0, 0.2, 0.5, 0.0, 0.5, 1.0]],
                [0, [LATTICE_SYSTEM_INDEX[TRICLINIC]], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
            ],
            [
                [1.0, 1.0, 1.0, 30.0, 30.0, 30.0],
                [1.0, 1.8, 3.0, 30.0, 90.0, 150.0],
                [5.0, 5.0, 5.0, 150.0, 150.0, 150.0],
                [1.0, 1.0, 1.0, 30.0, 30.0, 30.0],
                [1.0, 1.8, 3.0, 30.0, 90.0, 150.0],
                [5.0, 5.0, 5.0, 150.0, 150.0, 150.0],
            ],
        ),
        (
            CUBIC,
            [
                [1, [LATTICE_SYSTEM_INDEX[CUBIC]], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                [1, [LATTICE_SYSTEM_INDEX[CUBIC]], [0.25, 0.5, 0.75, 0.25, 0.5, 0.75]],
                [1, [LATTICE_SYSTEM_INDEX[CUBIC]], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
                [0, [LATTICE_SYSTEM_INDEX[CUBIC]], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                [0, [LATTICE_SYSTEM_INDEX[CUBIC]], [0.25, 0.5, 0.75, 0.25, 0.5, 0.75]],
                [0, [LATTICE_SYSTEM_INDEX[CUBIC]], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
            ],
            [
                [1.0, 1.0, 1.0, 90.0, 90.0, 90.0],
                [2.0, 2.0, 2.0, 90.0, 90.0, 90.0],
                [5.0, 5.0, 5.0, 90.0, 90.0, 90.0],
                [1.0, 1.0, 1.0, 90.0, 90.0, 90.0],
                [2.0, 2.0, 2.0, 90.0, 90.0, 90.0],
                [5.0, 5.0, 5.0, 90.0, 90.0, 90.0],
            ],
        ),
        (
            TRICLINIC,
            [
                [1, [LATTICE_SYSTEM_INDEX[CUBIC]], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                [1, [LATTICE_SYSTEM_INDEX[CUBIC]], [0.25, 0.5, 0.75, 0.25, 0.5, 0.75]],
                [1, [LATTICE_SYSTEM_INDEX[CUBIC]], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
                [0, [LATTICE_SYSTEM_INDEX[CUBIC]], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                [0, [LATTICE_SYSTEM_INDEX[CUBIC]], [0.25, 0.5, 0.75, 0.25, 0.5, 0.75]],
                [0, [LATTICE_SYSTEM_INDEX[CUBIC]], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
            ],
            [
                [1.0, 1.0, 1.0, 90.0, 90.0, 90.0],
                [2.0, 2.0, 2.0, 90.0, 90.0, 90.0],
                [5.0, 5.0, 5.0, 90.0, 90.0, 90.0],
                [1.0, 1.0, 1.0, 90.0, 90.0, 90.0],
                [2.0, 2.0, 2.0, 90.0, 90.0, 90.0],
                [5.0, 5.0, 5.0, 90.0, 90.0, 90.0],
            ],
        ),
        (
            TRICLINIC,
            [
                [1, [LATTICE_SYSTEM_INDEX[CUBIC]], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                [1, [LATTICE_SYSTEM_INDEX[CUBIC]], [0.25, 0.5, 0.75, 0.25, 0.5, 0.75]],
                [1, [LATTICE_SYSTEM_INDEX[CUBIC]], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
                [1, [LATTICE_SYSTEM_INDEX[HEXAGONAL]], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                [
                    1,
                    [LATTICE_SYSTEM_INDEX[HEXAGONAL]],
                    [0.25, 0.5, 0.75, 0.25, 0.5, 0.75],
                ],
                [1, [LATTICE_SYSTEM_INDEX[HEXAGONAL]], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
                [1, [LATTICE_SYSTEM_INDEX[MONOCLINIC]], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                [
                    1,
                    [LATTICE_SYSTEM_INDEX[MONOCLINIC]],
                    [0.25, 0.5, 0.75, 0.25, 0.5, 0.75],
                ],
                [1, [LATTICE_SYSTEM_INDEX[MONOCLINIC]], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
                [
                    1,
                    [LATTICE_SYSTEM_INDEX[ORTHORHOMBIC]],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                [
                    1,
                    [LATTICE_SYSTEM_INDEX[ORTHORHOMBIC]],
                    [0.25, 0.5, 0.75, 0.25, 0.5, 0.75],
                ],
                [
                    1,
                    [LATTICE_SYSTEM_INDEX[ORTHORHOMBIC]],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                ],
                [
                    1,
                    [LATTICE_SYSTEM_INDEX[RHOMBOHEDRAL]],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                [
                    1,
                    [LATTICE_SYSTEM_INDEX[RHOMBOHEDRAL]],
                    [0.25, 0.5, 0.75, 0.25, 0.5, 0.75],
                ],
                [
                    1,
                    [LATTICE_SYSTEM_INDEX[RHOMBOHEDRAL]],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                ],
                [1, [LATTICE_SYSTEM_INDEX[TETRAGONAL]], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                [
                    1,
                    [LATTICE_SYSTEM_INDEX[TETRAGONAL]],
                    [0.25, 0.5, 0.75, 0.25, 0.5, 0.75],
                ],
                [1, [LATTICE_SYSTEM_INDEX[TETRAGONAL]], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
                [1, [LATTICE_SYSTEM_INDEX[TRICLINIC]], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                [
                    1,
                    [LATTICE_SYSTEM_INDEX[TRICLINIC]],
                    [0.25, 0.5, 0.75, 0.25, 0.5, 0.75],
                ],
                [1, [LATTICE_SYSTEM_INDEX[TRICLINIC]], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
            ],
            [
                # CUBIC
                [1.0, 1.0, 1.0, 90.0, 90.0, 90.0],
                [2.0, 2.0, 2.0, 90.0, 90.0, 90.0],
                [5.0, 5.0, 5.0, 90.0, 90.0, 90.0],
                # HEXAGONAL
                [1.0, 1.0, 1.0, 90.0, 90.0, 120.0],
                [2.0, 2.0, 4.0, 90.0, 90.0, 120.0],
                [5.0, 5.0, 5.0, 90.0, 90.0, 120.0],
                # MONOCLINIC
                [1.0, 1.0, 1.0, 90.0, 30.0, 90.0],
                [2.0, 3.0, 4.0, 90.0, 90.0, 90.0],
                [5.0, 5.0, 5.0, 90.0, 150.0, 90.0],
                # ORTHORHOMBIC
                [1.0, 1.0, 1.0, 90.0, 90.0, 90.0],
                [2.0, 3.0, 4.0, 90.0, 90.0, 90.0],
                [5.0, 5.0, 5.0, 90.0, 90.0, 90.0],
                # RHOMBOHEDRAL
                [1.0, 1.0, 1.0, 30.0, 30.0, 30.0],
                [2.0, 2.0, 2.0, 60.0, 60.0, 60.0],
                [5.0, 5.0, 5.0, 150.0, 150.0, 150.0],
                # TETRAGONAL
                [1.0, 1.0, 1.0, 90.0, 90.0, 90.0],
                [2.0, 2.0, 4.0, 90.0, 90.0, 90.0],
                [5.0, 5.0, 5.0, 90.0, 90.0, 90.0],
                # TRICLINIC
                [1.0, 1.0, 1.0, 30.0, 30.0, 30.0],
                [2.0, 3.0, 4.0, 60.0, 90.0, 120.0],
                [5.0, 5.0, 5.0, 150.0, 150.0, 150.0],
            ],
        ),
        (
            CUBIC,
            [
                [1, [LATTICE_SYSTEM_INDEX[CUBIC]], [0.25, 0.5, 0.75, 0.25, 0.5, 0.75]],
            ],
            [
                [2.0, 2.0, 2.0, 90.0, 90.0, 90.0],
            ],
        ),
    ],
)
def test__states2proxy__returns_expected(env, lattice_system, states, expected):
    """
    Various lattice systems are tried because the conversion should be independent of
    the lattice system, since the states include the lattice system.
    """
    assert torch.equal(env.states2proxy(states), torch.tensor(expected))


@pytest.mark.parametrize(
    "lattice_system, states, expected",
    [
        (
            TRICLINIC,
            [
                [1, [LATTICE_SYSTEM_INDEX[TRICLINIC]], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                [1, [LATTICE_SYSTEM_INDEX[TRICLINIC]], [0.0, 0.2, 0.5, 0.0, 0.5, 1.0]],
                [1, [LATTICE_SYSTEM_INDEX[TRICLINIC]], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
                [0, [LATTICE_SYSTEM_INDEX[TRICLINIC]], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                [0, [LATTICE_SYSTEM_INDEX[TRICLINIC]], [0.0, 0.2, 0.5, 0.0, 0.5, 1.0]],
                [0, [LATTICE_SYSTEM_INDEX[TRICLINIC]], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
            ],
            [
                [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
                [-1.0, -0.6, 0.0, -1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
                [-1.0, -0.6, 0.0, -1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ],
        ),
        (
            CUBIC,
            [
                [1, [LATTICE_SYSTEM_INDEX[TRICLINIC]], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                [1, [LATTICE_SYSTEM_INDEX[TRICLINIC]], [0.0, 0.2, 0.5, 0.0, 0.5, 1.0]],
                [1, [LATTICE_SYSTEM_INDEX[TRICLINIC]], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
                [0, [LATTICE_SYSTEM_INDEX[TRICLINIC]], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                [0, [LATTICE_SYSTEM_INDEX[TRICLINIC]], [0.0, 0.2, 0.5, 0.0, 0.5, 1.0]],
                [0, [LATTICE_SYSTEM_INDEX[TRICLINIC]], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
            ],
            [
                [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
                [-1.0, -0.6, 0.0, -1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
                [-1.0, -0.6, 0.0, -1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ],
        ),
        (
            CUBIC,
            [
                [1, [LATTICE_SYSTEM_INDEX[CUBIC]], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                [1, [LATTICE_SYSTEM_INDEX[CUBIC]], [0.25, 0.5, 0.75, 0.25, 0.5, 0.75]],
                [1, [LATTICE_SYSTEM_INDEX[CUBIC]], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
                [0, [LATTICE_SYSTEM_INDEX[CUBIC]], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                [0, [LATTICE_SYSTEM_INDEX[CUBIC]], [0.25, 0.5, 0.75, 0.25, 0.5, 0.75]],
                [0, [LATTICE_SYSTEM_INDEX[CUBIC]], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
            ],
            [
                [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
                [-0.5, 0.0, 0.5, -0.5, 0.0, 0.5],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
                [-0.5, 0.0, 0.5, -0.5, 0.0, 0.5],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ],
        ),
        (
            TRICLINIC,
            [
                [1, [LATTICE_SYSTEM_INDEX[CUBIC]], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                [1, [LATTICE_SYSTEM_INDEX[CUBIC]], [0.25, 0.5, 0.75, 0.25, 0.5, 0.75]],
                [1, [LATTICE_SYSTEM_INDEX[CUBIC]], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
                [0, [LATTICE_SYSTEM_INDEX[CUBIC]], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                [0, [LATTICE_SYSTEM_INDEX[CUBIC]], [0.25, 0.5, 0.75, 0.25, 0.5, 0.75]],
                [0, [LATTICE_SYSTEM_INDEX[CUBIC]], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
            ],
            [
                [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
                [-0.5, 0.0, 0.5, -0.5, 0.0, 0.5],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
                [-0.5, 0.0, 0.5, -0.5, 0.0, 0.5],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ],
        ),
        (
            TRICLINIC,
            [
                [1, [LATTICE_SYSTEM_INDEX[CUBIC]], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                [1, [LATTICE_SYSTEM_INDEX[CUBIC]], [0.25, 0.5, 0.75, 0.25, 0.5, 0.75]],
                [1, [LATTICE_SYSTEM_INDEX[CUBIC]], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
                [1, [LATTICE_SYSTEM_INDEX[HEXAGONAL]], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                [
                    1,
                    [LATTICE_SYSTEM_INDEX[HEXAGONAL]],
                    [0.25, 0.5, 0.75, 0.25, 0.5, 0.75],
                ],
                [1, [LATTICE_SYSTEM_INDEX[HEXAGONAL]], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
                [1, [LATTICE_SYSTEM_INDEX[MONOCLINIC]], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                [
                    1,
                    [LATTICE_SYSTEM_INDEX[MONOCLINIC]],
                    [0.25, 0.5, 0.75, 0.25, 0.5, 0.75],
                ],
                [1, [LATTICE_SYSTEM_INDEX[MONOCLINIC]], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
                [
                    1,
                    [LATTICE_SYSTEM_INDEX[ORTHORHOMBIC]],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                [
                    1,
                    [LATTICE_SYSTEM_INDEX[ORTHORHOMBIC]],
                    [0.25, 0.5, 0.75, 0.25, 0.5, 0.75],
                ],
                [
                    1,
                    [LATTICE_SYSTEM_INDEX[ORTHORHOMBIC]],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                ],
                [
                    1,
                    [LATTICE_SYSTEM_INDEX[RHOMBOHEDRAL]],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                [
                    1,
                    [LATTICE_SYSTEM_INDEX[RHOMBOHEDRAL]],
                    [0.25, 0.5, 0.75, 0.25, 0.5, 0.75],
                ],
                [
                    1,
                    [LATTICE_SYSTEM_INDEX[RHOMBOHEDRAL]],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                ],
                [1, [LATTICE_SYSTEM_INDEX[TETRAGONAL]], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                [
                    1,
                    [LATTICE_SYSTEM_INDEX[TETRAGONAL]],
                    [0.25, 0.5, 0.75, 0.25, 0.5, 0.75],
                ],
                [1, [LATTICE_SYSTEM_INDEX[TETRAGONAL]], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
                [1, [LATTICE_SYSTEM_INDEX[TRICLINIC]], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                [
                    1,
                    [LATTICE_SYSTEM_INDEX[TRICLINIC]],
                    [0.25, 0.5, 0.75, 0.25, 0.5, 0.75],
                ],
                [1, [LATTICE_SYSTEM_INDEX[TRICLINIC]], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
            ],
            [
                # CUBIC
                [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
                [-0.5, 0.0, 0.5, -0.5, 0.0, 0.5],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                # HEXAGONAL
                [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
                [-0.5, 0.0, 0.5, -0.5, 0.0, 0.5],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                # MONOCLINIC
                [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
                [-0.5, 0.0, 0.5, -0.5, 0.0, 0.5],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                # ORTHORHOMBIC
                [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
                [-0.5, 0.0, 0.5, -0.5, 0.0, 0.5],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                # RHOMBOHEDRAL
                [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
                [-0.5, 0.0, 0.5, -0.5, 0.0, 0.5],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                # TETRAGONAL
                [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
                [-0.5, 0.0, 0.5, -0.5, 0.0, 0.5],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                # TRICLINIC
                [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
                [-0.5, 0.0, 0.5, -0.5, 0.0, 0.5],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ],
        ),
    ],
)
def test__states2policy__returns_expected(env, lattice_system, states, expected):
    """
    Various lattice systems are tried because the conversion should be independent of
    the lattice system, since the states include the lattice system.
    """
    assert torch.equal(env.states2policy(states), torch.tensor(expected))


@pytest.mark.parametrize(
    "lattice_system, expected_output",
    [
        (CUBIC, "Stage 0; cubic; (-1.0, -1.0, -1.0), (90.0, 90.0, 90.0)"),
        (HEXAGONAL, "Stage 0; hexagonal; (-1.0, -1.0, -1.0), (90.0, 90.0, 120.0)"),
        (MONOCLINIC, "Stage 0; monoclinic; (-1.0, -1.0, -1.0), (90.0, -1.0, 90.0)"),
        (ORTHORHOMBIC, "Stage 0; orthorhombic; (-1.0, -1.0, -1.0), (90.0, 90.0, 90.0)"),
        (RHOMBOHEDRAL, "Stage 0; rhombohedral; (-1.0, -1.0, -1.0), (-1.0, -1.0, -1.0)"),
        (TETRAGONAL, "Stage 0; tetragonal; (-1.0, -1.0, -1.0), (90.0, 90.0, 90.0)"),
        (TRICLINIC, "Stage 0; triclinic; (-1.0, -1.0, -1.0), (-1.0, -1.0, -1.0)"),
    ],
)
def test__state2readable__gives_expected_results_for_source_states(
    env, lattice_system, expected_output
):
    assert env.state2readable() == expected_output


@pytest.mark.parametrize(
    "lattice_system, readable",
    [
        (CUBIC, "Stage 0; cubic; (-1.0, -1.0, -1.0), (90.0, 90.0, 90.0)"),
        (HEXAGONAL, "Stage 0; hexagonal; (-1.0, -1.0, -1.0), (90.0, 90.0, 120.0)"),
        (MONOCLINIC, "Stage 0; monoclinic; (-1.0, -1.0, -1.0), (90.0, -1.0, 90.0)"),
        (ORTHORHOMBIC, "Stage 0; orthorhombic; (-1.0, -1.0, -1.0), (90.0, 90.0, 90.0)"),
        (RHOMBOHEDRAL, "Stage 0; rhombohedral; (-1.0, -1.0, -1.0), (-1.0, -1.0, -1.0)"),
        (TETRAGONAL, "Stage 0; tetragonal; (-1.0, -1.0, -1.0), (90.0, 90.0, 90.0)"),
        (TRICLINIC, "Stage 0; triclinic; (-1.0, -1.0, -1.0), (-1.0, -1.0, -1.0)"),
    ],
)
def test__readable2state__gives_expected_results_for_source_states(
    env, lattice_system, readable
):
    assert env.readable2state(readable) == env.source


@pytest.mark.parametrize(
    "lattice_system, state_init, action, state_expected",
    [
        (
            TETRAGONAL,
            [1, [LATTICE_SYSTEM_INDEX[TETRAGONAL]], [0.1, 0.0, 0.3, 0.0, 0.0, 0.0]],
            (1, 0.1, 0.0, 0.3, 0.0, 0.0, 0.0, 1),
            [1, [LATTICE_SYSTEM_INDEX[TETRAGONAL]], [-1, -1, -1, -1, -1, -1]],
        ),
        (
            TETRAGONAL,
            [1, [LATTICE_SYSTEM_INDEX[TETRAGONAL]], [-1, -1, -1, -1, -1, -1]],
            (0, 0, 0, 0, 0, 0, 0, 0),
            [0, [LATTICE_SYSTEM_INDEX[TETRAGONAL]], [-1, -1, -1, -1, -1, -1]],
        ),
    ],
)
def test__step_backwards__behaves_as_expected(
    env, lattice_system, state_init, action, state_expected
):
    env.set_state(state_init)
    assert env.equal(env.state, state_init)
    env.step_backwards(action)
    assert env.equal(env.state, state_expected)


@pytest.mark.parametrize(
    "lattice_system",
    [CUBIC, HEXAGONAL, MONOCLINIC, ORTHORHOMBIC, RHOMBOHEDRAL, TETRAGONAL, TRICLINIC],
)
class TestContinuousLatticeBasic(common.BaseTestsContinuous):
    @pytest.fixture(autouse=True)
    def setup(self, env, lattice_system):
        self.env = env  # lattice_system intializes env fixture.
        self.repeats = {
            "test__reset__state_is_source": 10,
            "test__forward_actions_have_nonzero_backward_prob": 10,
            "test__backward_actions_have_nonzero_forward_prob": 10,
            "test__trajectories_are_reversible": 10,
            "test__step_random__does_not_sample_invalid_actions_forward": 10,
            "test__step_random__does_not_sample_invalid_actions_backward": 10,
            "test__sample_actions__get_logprobs__return_valid_actions_and_logprobs": 10,
            "test__get_parents_step_get_mask__are_compatible": 10,
            "test__sample_backwards_reaches_source": 10,
            "test__state2readable__is_reversible": 20,
            "test__gflownet_minimal_runs": 3,
        }
        self.n_states = {
            "test__backward_actions_have_nonzero_forward_prob": 10,
            "test__sample_backwards_reaches_source": 10,
            "test__get_logprobs__all_finite_in_random_forward_transitions": 10,
            "test__get_logprobs__all_finite_in_random_backward_transitions": 10,
        }
        self.batch_size = {
            "test__sample_actions__get_logprobs__batched_forward_trajectories": 10,
            "test__sample_actions__get_logprobs__batched_backward_trajectories": 10,
            "test__get_logprobs__all_finite_in_accumulated_forward_trajectories": 10,
            "test__gflownet_minimal_runs": 10,
        }
