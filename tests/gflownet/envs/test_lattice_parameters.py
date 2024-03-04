"""
These tests are for the continuous lattice parameters environment. The tests for the
former discrete implementation have been removed for simplicity.  Check commit
9f3477d8e46c4624f9162d755663993b83196546 to see these changes or the history previous
to that commit to consult previous implementations.
"""

import common
import pytest
import torch

from gflownet.envs.crystals.lattice_parameters import PARAMETER_NAMES, LatticeParameters
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


@pytest.mark.parametrize(
    "lattice_system",
    [RHOMBOHEDRAL],
)
@pytest.mark.repeat(N_REPETITIONS)
def test__rhombohedral__constraints_remain_after_random_actions(env, lattice_system):
    env = env.reset()
    while not env.done:
        env.step_random()
        (a, b, c), (alpha, beta, gamma) = env._unpack_lengths_angles()
        assert len({a, b, c}) == 1
        assert len({alpha, beta, gamma}) == 1
        assert len({alpha, beta, gamma, 90.0}) == 2


@pytest.mark.parametrize(
    "lattice_system",
    [TETRAGONAL],
)
@pytest.mark.repeat(N_REPETITIONS)
def test__tetragonal__constraints_remain_after_random_actions(env, lattice_system):
    env = env.reset()
    while not env.done:
        env.step_random()
        (a, b, c), (alpha, beta, gamma) = env._unpack_lengths_angles()
        assert a == b
        assert len({a, b, c}) == 2
        assert len({alpha, beta, gamma, 90.0}) == 1


@pytest.mark.parametrize(
    "lattice_system",
    [TRICLINIC],
)
@pytest.mark.repeat(N_REPETITIONS)
def test__triclinic__constraints_remain_after_random_actions(env, lattice_system):
    env = env.reset()
    while not env.done:
        env.step_random()
        (a, b, c), (alpha, beta, gamma) = env._unpack_lengths_angles()
        assert len({a, b, c}) == 3
        assert len({alpha, beta, gamma, 90.0}) == 4


@pytest.mark.parametrize(
    "lattice_system, states, expected",
    [
        (
            TRICLINIC,
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.2, 0.5, 0.0, 0.5, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ],
            [
                [1.0, 1.0, 1.0, 30.0, 30.0, 30.0],
                [1.0, 1.8, 3.0, 30.0, 90.0, 150.0],
                [5.0, 5.0, 5.0, 150.0, 150.0, 150.0],
            ],
        ),
        (
            CUBIC,
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.25, 0.5, 0.75, 0.25, 0.5, 0.75],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ],
            [
                [1.0, 1.0, 1.0, 30.0, 30.0, 30.0],
                [2.0, 3.0, 4.0, 60.0, 90.0, 120.0],
                [5.0, 5.0, 5.0, 150.0, 150.0, 150.0],
            ],
        ),
    ],
)
def test__states2proxy__returns_expected(env, lattice_system, states, expected):
    """
    Various lattice systems are tried because the conversion should be independent of
    the lattice system, since the states are expected to satisfy the constraints.
    """
    assert torch.equal(env.states2proxy(states), torch.tensor(expected))


@pytest.mark.parametrize(
    "lattice_system, states, expected",
    [
        (
            TRICLINIC,
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.2, 0.5, 0.0, 0.5, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ],
            [
                [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
                [-1.0, -0.6, 0.0, -1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ],
        ),
        (
            CUBIC,
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.25, 0.5, 0.75, 0.25, 0.5, 0.75],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ],
            [
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
    the lattice system, since the states are expected to satisfy the constraints.
    """
    assert torch.equal(env.states2policy(states), torch.tensor(expected))


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
@pytest.mark.skip(reason="skip until it gets updated")
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
@pytest.mark.skip(reason="skip until it gets updated")
def test__readable2state__gives_expected_results_for_initial_states(
    env, lattice_system, readable
):
    assert env.readable2state(readable) == env.state


@pytest.mark.parametrize(
    "lattice_system",
    [CUBIC, HEXAGONAL, MONOCLINIC, ORTHORHOMBIC, RHOMBOHEDRAL, TETRAGONAL, TRICLINIC],
)
class TestContinuousLatticeBasic(common.BaseTestsContinuous):
    @pytest.fixture(autouse=True)
    def setup(self, env, lattice_system):
        self.env = env  # lattice_system intializes env fixture.
        self.repeats = {
            "test__get_logprobs__backward__returns_zero_if_done": 100,  # Overrides no repeat.
            "test__reset__state_is_source": 10,
        }
        self.n_states = {}  # TODO: Populate.
