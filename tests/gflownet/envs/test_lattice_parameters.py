import pytest

import torch

from gflownet.envs.lattice_parameters import (
    CUBIC,
    HEXAGONAL,
    LATTICE_SYSTEMS,
    LatticeParameters,
    MONOCLINIC,
    ORTHORHOMBIC,
    RHOMBOHEDRAL,
    TETRAGONAL,
    TRICLINIC,
)


@pytest.fixture()
def env(lattice_system):
    return LatticeParameters(lattice_system=lattice_system)


@pytest.mark.parametrize("lattice_system", LATTICE_SYSTEMS)
def test__environment__initializes_properly(env, lattice_system):
    pass


@pytest.mark.parametrize("lattice_system", LATTICE_SYSTEMS)
@pytest.mark.parametrize("max_step_len", [1, 3, 5])
def test__get_actions_space__returns_correct_number_of_actions(
    lattice_system, max_step_len
):
    environment = LatticeParameters(
        lattice_system=lattice_system, max_step_len=max_step_len
    )
    exp_n_actions = 9 * max_step_len + 1

    assert len(environment.get_actions_space()) == exp_n_actions


@pytest.mark.parametrize("lattice_system", [CUBIC, HEXAGONAL, ORTHORHOMBIC, TETRAGONAL])
def test__get_mask_invalid_actions_forward__disallows_angle_change_for_selected_systems(
    env, lattice_system
):
    mask = env.get_mask_invalid_actions_forward()
    angle_mask = mask[5:9]

    assert all(angle_mask)


@pytest.mark.parametrize("lattice_system", [TRICLINIC])
def test__get_mask_invalid_actions_forward__allows_arbitrary_angle_change_for_triclinic(
    env, lattice_system
):
    mask = env.get_mask_invalid_actions_forward()
    angle_mask = mask[5:9]

    assert not any(angle_mask)


@pytest.mark.parametrize("lattice_system", LATTICE_SYSTEMS)
def test__get_parents__returns_no_parents_in_initial_state(env, lattice_system):
    parents, actions = env.get_parents()

    assert len(parents) == 0
    assert len(actions) == 0


@pytest.mark.parametrize("lattice_system", LATTICE_SYSTEMS)
def test__get_parents__returns_parents_after_step(env, lattice_system):
    env.step((0, 1, 2))

    parents, actions = env.get_parents()

    assert len(parents) != 0
    assert len(actions) != 0


@pytest.mark.parametrize("lattice_system", LATTICE_SYSTEMS)
@pytest.mark.parametrize(
    "actions",
    [[], [(1, 2), (2, 3), (3, 4)], [(4, 2)], [(1, 3), (4, 2), (2, 3), (3, 2)]],
)
def test__get_parents__returns_same_number_of_parents_and_actions(
    env, lattice_system, actions
):
    for action in actions:
        env.step(action=action)

    parents, actions = env.get_parents()

    assert len(parents) == len(actions)


@pytest.mark.parametrize(
    "lattice_system, state, exp_tensor",
    [
        (
            TRICLINIC,
            [0, 0, 0, 0, 0, 0],
            [1.0, 1.0, 1.0, 30.0, 30.0, 30.0],
        ),
        (
            TRICLINIC,
            [60, 60, 60, 60, 60, 60],
            [5.0, 5.0, 5.0, 150.0, 150.0, 150.0],
        ),
    ],
)
def test__state2oracle__returns_expected_tensor(env, lattice_system, state, exp_tensor):
    assert torch.equal(env.state2oracle(state), torch.Tensor(exp_tensor))


@pytest.mark.parametrize("lattice_system", [TRICLINIC])
def test__reset(env, lattice_system):
    env.step((0, 1, 2))

    assert env.state != env.source

    env.reset()

    assert env.state == env.source


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
        (RHOMBOHEDRAL, "(1.0, 1.0, 1.0), (30.0, 30.0, 30.0)"),
        (TRICLINIC, "(1.0, 1.0, 1.0), (30.0, 30.0, 30.0)"),
    ],
)
def test__readable2state__returns_initial_state_for_rhombohedral_and_triclinic(
    env, lattice_system, readable
):
    assert env.readable2state(readable) == [0, 0, 0, 0, 0, 0]
