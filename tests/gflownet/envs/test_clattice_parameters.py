import common
import pytest
import torch

from gflownet.envs.crystals.clattice_parameters import (
    CUBIC,
    HEXAGONAL,
    MONOCLINIC,
    ORTHORHOMBIC,
    PARAMETER_NAMES,
    RHOMBOHEDRAL,
    TETRAGONAL,
    TRICLINIC,
    CLatticeParameters,
)
from gflownet.envs.crystals.lattice_parameters import LATTICE_SYSTEMS

N_REPETITIONS = 1000


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
        (CUBIC, [None, None, None, 90, 90, 90]),
        (HEXAGONAL, [None, None, None, 90, 90, 120]),
        (MONOCLINIC, [None, None, None, 90, None, 90]),
        (ORTHORHOMBIC, [None, None, None, 90, 90, 90]),
        (RHOMBOHEDRAL, [None, None, None, None, None, None]),
        (TETRAGONAL, [None, None, None, 90, 90, 90]),
        (TRICLINIC, [None, None, None, None, None, None]),
    ],
)
def test__environment__has_expected_fixed_parameters(
    env, lattice_system, expected_params
):
    for expected_value, param_name in zip(expected_params, PARAMETER_NAMES):
        if expected_value is not None:
            assert getattr(env, param_name) == expected_value


@pytest.mark.parametrize(
    "lattice_system",
    [CUBIC],
)
@pytest.mark.repeat(N_REPETITIONS)
def test__cubic__constraints_remain_after_random_actions(env, lattice_system):
    env = env.reset()
    while not env.done:
        (a, b, c), (alpha, beta, gamma) = env._unpack_lengths_angles()
        assert len({a, b, c}) == 1
        assert len({alpha, beta, gamma, 90.0}) == 1
        env.step_random()


@pytest.mark.parametrize(
    "lattice_system",
    [HEXAGONAL],
)
@pytest.mark.repeat(N_REPETITIONS)
def test__hexagonal__constraints_remain_after_random_actions(env, lattice_system):
    env = env.reset()
    while not env.done:
        env.step_random()
        (a, b, c), (alpha, beta, gamma) = env._unpack_lengths_angles()
        assert a == b
        assert len({a, b, c}) == 2
        assert len({alpha, beta, 90.0}) == 1
        assert gamma == 120.0


@pytest.mark.parametrize(
    "lattice_system",
    [MONOCLINIC],
)
@pytest.mark.repeat(N_REPETITIONS)
def test__monoclinic__constraints_remain_after_random_actions(env, lattice_system):
    env = env.reset()
    while not env.done:
        env.step_random()
        (a, b, c), (alpha, beta, gamma) = env._unpack_lengths_angles()
        assert len({a, b, c}) == 3
        assert len({alpha, gamma, 90.0}) == 1
        assert beta != 90.0


@pytest.mark.parametrize(
    "lattice_system",
    [ORTHORHOMBIC],
)
@pytest.mark.repeat(N_REPETITIONS)
def test__orthorhombic__constraints_remain_after_random_actions(env, lattice_system):
    env = env.reset()
    while not env.done:
        env.step_random()
        (a, b, c), (alpha, beta, gamma) = env._unpack_lengths_angles()
        assert len({a, b, c}) == 3
        assert len({alpha, beta, gamma, 90.0}) == 1


@pytest.mark.parametrize("lattice_system", [TRICLINIC])
@pytest.mark.parametrize(
    "state, expected_output",
    [
        (
            [0, 0, 0, 0, 0, 0],
            torch.Tensor([1.0, 1.0, 1.0, 30.0, 30.0, 30.0]),
        ),
        (
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            torch.Tensor([1.4, 1.8, 2.2, 78.0, 90.0, 102.0]),
        ),
    ],
)
def test__state2proxy__has_expected_output(env, lattice_system, state, expected_output):
    assert torch.allclose(env.state2proxy(state), expected_output)


@pytest.mark.parametrize("lattice_system", [TRICLINIC])
@pytest.mark.parametrize(
    "states, expected_output",
    [
        (
            [[0, 0, 0, 0, 0, 0], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]],
            torch.Tensor(
                [[1.0, 1.0, 1.0, 30.0, 30.0, 30.0], [1.4, 1.8, 2.2, 78.0, 90.0, 102.0]]
            ),
        ),
    ],
)
def test__statetorch2proxy__has_expected_output(
    env, lattice_system, states, expected_output
):
    assert torch.allclose(env.statebatch2policy(states), expected_output)
