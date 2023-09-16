import common
import pytest
import torch

from gflownet.envs.crystals.clattice_parameters import (
    CUBIC,
    HEXAGONAL,
    LATTICE_SYSTEMS,
    MONOCLINIC,
    ORTHORHOMBIC,
    RHOMBOHEDRAL,
    TETRAGONAL,
    TRICLINIC,
    CLatticeParameters,
)

N_REPETITIONS = 100


@pytest.fixture()
def env(lattice_system):
    return CLatticeParameters(
        lattice_system=lattice_system,
        min_length=1.0,
        max_length=5.0,
        min_angle=30.0,
        max_angle=150.0,
    )


@pytest.mark.parametrize("lattice_system", LATTICE_SYSTEMS)
def test__environment__initializes_properly(env, lattice_system):
    pass


@pytest.mark.parametrize(
    "lattice_system, expected_params",
    [
        (CUBIC, [1, 1, 1, 90, 90, 90]),
        (HEXAGONAL, [1, 1, 1, 90, 90, 120]),
        (MONOCLINIC, [1, 1, 1, 90, 30, 90]),
        (ORTHORHOMBIC, [1, 1, 1, 90, 90, 90]),
        (RHOMBOHEDRAL, [1, 1, 1, 30, 30, 30]),
        (TETRAGONAL, [1, 1, 1, 90, 90, 90]),
        (TRICLINIC, [1, 1, 1, 30, 30, 30]),
    ],
)
def test__environment__has_expected_initial_parameters(
    env, lattice_system, expected_params
):
    (a, b, c), (alpha, beta, gamma) = env._unpack_lengths_angles()
    assert a == expected_params[0]
    assert b == expected_params[1]
    assert c == expected_params[2]
    assert alpha == expected_params[3]
    assert beta == expected_params[4]
    assert gamma == expected_params[5]


@pytest.mark.parametrize(
    "lattice_system",
    [CUBIC],
)
@pytest.mark.repeat(N_REPETITIONS)
def test__cubic__constraints_remain_after_random_actions(env, lattice_system):
    env = env.reset()
    while not env.done:
        (a, b, c), (alpha, beta, gamma) = env._unpack_lengths_angles()
        assert a == b
        assert b == c
        assert a == c
        assert alpha == 90.0
        assert beta == 90.0
        assert gamma == 90.0
        env.step_random()


@pytest.mark.parametrize(
    "lattice_system",
    [HEXAGONAL],
)
@pytest.mark.repeat(N_REPETITIONS)
def test__hexagonal__constraints_remain_after_random_actions(env, lattice_system):
    env = env.reset()
    while not env.done:
        (a, b, c), (alpha, beta, gamma) = env._unpack_lengths_angles()
        assert a == b
        assert alpha == 90.0
        assert beta == 90.0
        assert gamma == 120.0
        env.step_random()


@pytest.mark.parametrize(
    "lattice_system",
    [MONOCLINIC],
)
@pytest.mark.repeat(N_REPETITIONS)
def test__monoclinic__constraints_remain_after_random_actions(env, lattice_system):
    env = env.reset()
    while not env.done:
        (a, b, c), (alpha, beta, gamma) = env._unpack_lengths_angles()
        assert alpha == 90.0
        assert beta != 90.0
        assert gamma == 90.0
        env.step_random()


@pytest.mark.parametrize(
    "lattice_system",
    [ORTHORHOMBIC],
)
@pytest.mark.repeat(N_REPETITIONS)
def test__orthorhombic__constraints_remain_after_random_actions(env, lattice_system):
    env = env.reset()
    while not env.done:
        (a, b, c), (alpha, beta, gamma) = env._unpack_lengths_angles()
        assert alpha == 90.0
        assert beta == 90.0
        assert gamma == 90.0
        env.step_random()


@pytest.mark.parametrize(
    "lattice_system",
    [RHOMBOHEDRAL],
)
@pytest.mark.repeat(N_REPETITIONS)
def test__rhombohedral__constraints_remain_after_random_actions(env, lattice_system):
    env = env.reset()
    while not env.done:
        (a, b, c), (alpha, beta, gamma) = env._unpack_lengths_angles()
        assert a == b
        assert b == c
        assert a == c
        assert alpha == beta
        assert beta == gamma
        assert alpha == gamma
        assert alpha != 90.0
        env.step_random()


@pytest.mark.parametrize(
    "lattice_system",
    [TETRAGONAL],
)
@pytest.mark.repeat(N_REPETITIONS)
def test__tetragonal__constraints_remain_after_random_actions(env, lattice_system):
    env = env.reset()
    while not env.done:
        (a, b, c), (alpha, beta, gamma) = env._unpack_lengths_angles()
        assert a == b
        assert alpha == 90.0
        assert beta == 90.0
        assert gamma == 90.0
        env.step_random()


@pytest.mark.parametrize(
    "lattice_system",
    [TRICLINIC],
)
@pytest.mark.repeat(N_REPETITIONS)
def test__triclinic__constraints_remain_after_random_actions(env, lattice_system):
    env = env.reset()
    while not env.done:
        # TODO: Test not equality constraints
        env.step_random()
        (a, b, c), (alpha, beta, gamma) = env._unpack_lengths_angles()
        assert len({alpha, beta, gamma, 90.0}) == 4


@pytest.mark.parametrize(
    "lattice_system, expected_output",
    [
        (CUBIC, "(1.0, 1.0, 1.0), (90.0, 90.0, 90.0)"),
        (HEXAGONAL, "(1.0, 1.0, 1.0), (90.0, 90.0, 120.0)"),
        (MONOCLINIC, "(1.0, 1.0, 1.0), (90.0, 30.0, 90.0)"),
        (ORTHORHOMBIC, "(1.0, 1.0, 1.0), (90.0, 90.0, 90.0)"),
        (RHOMBOHEDRAL, "(1.0, 1.0, 1.0), (30.0, 30.0, 30.0)"),
        (TETRAGONAL, "(1.0, 1.0, 1.0), (90.0, 90.0, 90.0)"),
        (TRICLINIC, "(1.0, 1.0, 1.0), (30.0, 30.0, 30.0)"),
    ],
)
def test__state2readable__gives_expected_results_for_initial_states(
    env, lattice_system, expected_output
):
    assert env.state2readable() == expected_output


@pytest.mark.parametrize(
    "lattice_system, readable",
    [
        (CUBIC, "(1.0, 1.0, 1.0), (90.0, 90.0, 90.0)"),
        (HEXAGONAL, "(1.0, 1.0, 1.0), (90.0, 90.0, 120.0)"),
        (MONOCLINIC, "(1.0, 1.0, 1.0), (90.0, 30.0, 90.0)"),
        (ORTHORHOMBIC, "(1.0, 1.0, 1.0), (90.0, 90.0, 90.0)"),
        (RHOMBOHEDRAL, "(1.0, 1.0, 1.0), (30.0, 30.0, 30.0)"),
        (TETRAGONAL, "(1.0, 1.0, 1.0), (90.0, 90.0, 90.0)"),
        (TRICLINIC, "(1.0, 1.0, 1.0), (30.0, 30.0, 30.0)"),
    ],
)
def test__readable2state__returns_initial_state_for_rhombohedral_and_triclinic(
    env, lattice_system, readable
):
    assert env.readable2state(readable) == env.state
