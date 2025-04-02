"""
These tests are for the continuous lattice parameters environment. The tests for the
former discrete implementation have been removed for simplicity.  Check commit
9f3477d8e46c4624f9162d755663993b83196546 to see these changes or the history previous
to that commit to consult previous implementations.
"""

import common
import numpy
import pytest
import torch

from gflownet.envs.crystals.lattice_parameters import (
    PARAMETER_NAMES,
    LatticeParametersSGCCG,
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
    return LatticeParametersSGCCG(
        lattice_system=lattice_system,
    )


@pytest.mark.parametrize("lattice_system", LATTICE_SYSTEMS)
def test__environment__initializes_properly(env, lattice_system):
    pass


@pytest.mark.parametrize(
    "lattice_system",
    [CUBIC],
)
@pytest.mark.repeat(N_REPETITIONS)
def test__cubic__constraints_remain_after_random_actions(env, lattice_system):
    env = env.reset()
    while not env.done:
        env.step_random()

        # Obtain the lattice parameters through state2lattice and ensure that the
        # lattice parameters implement the env's constraints exactly
        (a, b, c), (alpha, beta, gamma) = env.state2lattice()
        assert len({a, b, c}) == 1
        assert len({alpha, beta, gamma, 90.0}) == 1

        # Obtain the lattice parameters through _state2projection and
        # _projection2lattice which *don't* enforce the env's exact constraints on the
        # lattice parameters and ensure that the constraints are satisfied up to
        # rounding error
        state = env.state
        approx_lattice_params = env._projection2lattice(env._state2projection(state))
        _a, _b, _c, _alpha, _beta, _gamma = approx_lattice_params
        assert numpy.allclose([_a, _b], _c)
        assert numpy.allclose([_alpha, _beta, _gamma], 90.0)


@pytest.mark.parametrize(
    "lattice_system",
    [HEXAGONAL],
)
@pytest.mark.repeat(N_REPETITIONS)
def test__hexagonal__constraints_remain_after_random_actions(env, lattice_system):
    env = env.reset()
    while not env.done:
        env.step_random()

        # Obtain the lattice parameters through state2lattice and ensure that the
        # lattice parameters implement the env's constraints exactly
        (a, b, c), (alpha, beta, gamma) = env.state2lattice()
        assert a == b
        assert len({a, b, c}) == 2
        assert len({alpha, beta, 90.0}) == 1
        assert gamma == 120.0

        # Obtain the lattice parameters through _state2projection and
        # _projection2lattice which *don't* enforce the env's exact constraints on the
        # lattice parameters and ensure that the constraints are satisfied up to
        # rounding error
        state = env.state
        approx_lattice_params = env._projection2lattice(env._state2projection(state))
        _a, _b, _c, _alpha, _beta, _gamma = approx_lattice_params
        assert numpy.isclose(_a, _b)
        assert numpy.allclose([_alpha, _beta], 90.0)
        assert numpy.isclose(_gamma, 120.0)


@pytest.mark.parametrize(
    "lattice_system",
    [MONOCLINIC],
)
@pytest.mark.repeat(N_REPETITIONS)
def test__monoclinic__constraints_remain_after_random_actions(env, lattice_system):
    env = env.reset()
    while not env.done:
        env.step_random()

        # Obtain the lattice parameters through state2lattice and ensure that the
        # lattice parameters implement the env's constraints exactly
        (a, b, c), (alpha, beta, gamma) = env.state2lattice()
        assert len({a, b, c}) == 3
        assert len({alpha, gamma, 90.0}) == 1
        assert beta != 90.0

        # Obtain the lattice parameters through _state2projection and
        # _projection2lattice which *don't* enforce the env's exact constraints on the
        # lattice parameters and ensure that the constraints are satisfied up to
        # rounding error
        state = env.state
        approx_lattice_params = env._projection2lattice(env._state2projection(state))
        _a, _b, _c, _alpha, _beta, _gamma = approx_lattice_params
        assert numpy.allclose([_alpha, _gamma], 90.0)


@pytest.mark.parametrize(
    "lattice_system",
    [ORTHORHOMBIC],
)
@pytest.mark.repeat(N_REPETITIONS)
def test__orthorhombic__constraints_remain_after_random_actions(env, lattice_system):
    env = env.reset()
    while not env.done:
        env.step_random()

        # Obtain the lattice parameters through state2lattice and ensure that the
        # lattice parameters implement the env's constraints exactly
        (a, b, c), (alpha, beta, gamma) = env.state2lattice()
        assert len({a, b, c}) == 3
        assert len({alpha, beta, gamma, 90.0}) == 1

        # Obtain the lattice parameters through _state2projection and
        # _projection2lattice which *don't* enforce the env's exact constraints on the
        # lattice parameters and ensure that the constraints are satisfied up to
        # rounding error
        state = env.state
        approx_lattice_params = env._projection2lattice(env._state2projection(state))
        _a, _b, _c, _alpha, _beta, _gamma = approx_lattice_params
        assert numpy.allclose([_alpha, _beta, _gamma], 90.0)


@pytest.mark.parametrize(
    "lattice_system",
    [RHOMBOHEDRAL],
)
@pytest.mark.repeat(N_REPETITIONS)
def test__rhombohedral__constraints_remain_after_random_actions(env, lattice_system):
    env = env.reset()
    while not env.done:
        env.step_random()

        # Obtain the lattice parameters through state2lattice and ensure that the
        # lattice parameters implement the env's constraints exactly
        (a, b, c), (alpha, beta, gamma) = env.state2lattice()
        assert len({a, b, c}) == 1
        assert len({alpha, beta, gamma}) == 1
        assert len({alpha, beta, gamma, 90.0}) == 2

        # Obtain the lattice parameters through _state2projection and
        # _projection2lattice which *don't* enforce the env's exact constraints on the
        # lattice parameters and ensure that the constraints are satisfied up to
        # rounding error
        state = env.state
        approx_lattice_params = env._projection2lattice(env._state2projection(state))
        _a, _b, _c, _alpha, _beta, _gamma = approx_lattice_params
        assert numpy.allclose([_a, _b], _c)
        assert numpy.allclose([_alpha, _beta], _gamma)


@pytest.mark.parametrize(
    "lattice_system",
    [TETRAGONAL],
)
@pytest.mark.repeat(N_REPETITIONS)
def test__tetragonal__constraints_remain_after_random_actions(env, lattice_system):
    env = env.reset()
    while not env.done:
        env.step_random()

        # Obtain the lattice parameters through state2lattice and ensure that the
        # lattice parameters implement the env's constraints exactly
        (a, b, c), (alpha, beta, gamma) = env.state2lattice()
        assert a == b
        assert len({a, b, c}) == 2
        assert len({alpha, beta, gamma, 90.0}) == 1

        # Obtain the lattice parameters through _state2projection and
        # _projection2lattice which *don't* enforce the env's exact constraints on the
        # lattice parameters and ensure that the constraints are satisfied up to
        # rounding error
        state = env.state
        approx_lattice_params = env._projection2lattice(env._state2projection(state))
        _a, _b, _c, _alpha, _beta, _gamma = approx_lattice_params
        assert numpy.isclose(_a, _b)
        assert numpy.allclose([_alpha, _beta, _gamma], 90.0)


@pytest.mark.parametrize(
    "lattice_system",
    [TRICLINIC],
)
@pytest.mark.repeat(N_REPETITIONS)
def test__triclinic__constraints_remain_after_random_actions(env, lattice_system):
    env = env.reset()
    while not env.done:
        env.step_random()

        # Obtain the lattice parameters through state2lattice and ensure that the
        # lattice parameters implement the env's constraints exactly
        (a, b, c), (alpha, beta, gamma) = env.state2lattice()
        assert len({a, b, c}) == 3
        assert len({alpha, beta, gamma, 90.0}) == 4


@pytest.mark.parametrize(
    "lattice_system, states, expected",
    [
        (
            TRICLINIC,
            [
                [0.5, 0.5, 0.5, 0.5, 0.5, 3 / 17.0],
            ],
            [
                [1.0, 1.0, 1.0, 90.0, 90.0, 90.0],
            ],
        ),
        (
            CUBIC,
            [
                [0.5, 0.5, 0.5, 0.5, 0.5, 3 / 17.0],
            ],
            [
                [1.0, 1.0, 1.0, 90.0, 90.0, 90.0],
            ],
        ),
    ],
)
def test__states2proxy__returns_expected(env, lattice_system, states, expected):
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
    "lattice_system",
    [TRICLINIC],
)
def test__state2projection__mapping_invertible(env, lattice_system):

    # Sample states from the environment and, for each, ensure that is is possible to
    # map between the state space and the projection space without loss of information
    # (except for possible rounding error)
    terminating_states = env.get_grid_terminating_states(1000)
    for state in terminating_states:
        projection = env._state2projection(state)
        state_reconstructed = env._projection2state(projection)
        projection_reconstructed = env._state2projection(state_reconstructed)

        assert numpy.allclose(state, state_reconstructed)
        assert numpy.allclose(projection, projection_reconstructed)


@pytest.mark.parametrize(
    "lattice_system",
    [TRICLINIC],
)
def test__projection2lattice__mapping_invertible(env, lattice_system):

    # Sample states from the environment and, for each, ensure that is is possible to
    # map between the projection space and the lattice parameter space without loss of
    # information (except for possible rounding error)
    terminating_states = env.get_grid_terminating_states(1000)
    for state in terminating_states:
        projection = env._state2projection(state)

        lattice_params = env._projection2lattice(projection)
        projection_reconstructed = env._lattice2projection(lattice_params)
        lattice_params_reconstructed = env._projection2lattice(projection_reconstructed)

        assert numpy.allclose(projection, projection_reconstructed, atol=1e-4)
        assert numpy.allclose(lattice_params, lattice_params_reconstructed)


@pytest.mark.parametrize(
    "lattice_system",
    [TRICLINIC],
)
def test__env_can_represent_desired_lattice_params(env, lattice_system):

    # Sample states from the environment and, for each, ensure that is is possible to
    # map between the projection space and the lattice parameter space without loss of
    # information (except for possible rounding error)
    desired_lengths = [1, 10, 100, 1000]
    desired_angles = [10, 30, 60, 90, 120, 150, 170]

    from itertools import product

    for a, b, c in product(desired_lengths, repeat=3):
        for alpha, beta, gamma in product(desired_angles, repeat=3):
            lattice_params = [a, b, c, alpha, beta, gamma]

            # Skip any invalid combination of lattice parameters
            if not (0 < alpha + beta + gamma < 360):
                continue
            elif not (0 < -alpha + beta + gamma < 360):
                continue
            elif not (0 < alpha - beta + gamma < 360):
                continue
            elif not (0 < alpha + beta - gamma < 360):
                continue

            projection = env._lattice2projection(lattice_params)
            lattice_params_reconstructed = env._projection2lattice(projection)

            for proj_idx in range(len(projection)):
                assert projection[proj_idx] > env.min_projection_values[proj_idx]
                assert projection[proj_idx] < env.max_projection_values[proj_idx]

            # Ensure that the desired lattice parameters do not lie in a region of the
            # projection space where there might be numerical instability. Near-perfect
            # reconstruction of the lattice parameters is used as an indication that
            # the corresponding region in projection space is numerically stable.
            assert numpy.allclose(lattice_params, lattice_params_reconstructed)


@pytest.mark.parametrize(
    "lattice_system",
    [TRICLINIC],
)
def test__env_produces_valid_lattice_params(env, lattice_system):

    # Sample random trajectories from the environment and ensure that every intermediate
    # state represents valid lattice parameters
    env = env.reset()
    while not env.done:
        env.step_random()

        # Obtain the lattice parameters and ensure that they all have valid values
        (a, b, c), (alpha, beta, gamma) = env.state2lattice()
        assert min(a, b, c) > 0
        assert min(alpha, beta, gamma) > 0
        assert max(alpha, beta, gamma) < 360
        assert 0 < alpha + beta + gamma < 360
        assert 0 < -alpha + beta + gamma < 360
        assert 0 < alpha - beta + gamma < 360
        assert 0 < alpha + beta - gamma < 360


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
