import common
import pytest
import torch

from gflownet.envs.crystals.lattice_parameters import (
    CUBIC,
    HEXAGONAL,
    LATTICE_SYSTEMS,
    MONOCLINIC,
    ORTHORHOMBIC,
    RHOMBOHEDRAL,
    TETRAGONAL,
    TRICLINIC,
    LatticeParameters,
)


@pytest.fixture()
def env(lattice_system):
    return LatticeParameters(lattice_system=lattice_system, grid_size=61)


@pytest.fixture()
def triclinic_env():
    return LatticeParameters(lattice_system=TRICLINIC, grid_size=5)


@pytest.mark.parametrize("lattice_system", LATTICE_SYSTEMS)
def test__environment__initializes_properly(env, lattice_system):
    pass


@pytest.mark.parametrize("lattice_system", LATTICE_SYSTEMS)
@pytest.mark.parametrize("grid_size", [61, 121, 241])
def test__environment__initializes_properly_with_non_default_grid_size(
    env, lattice_system, grid_size
):
    LatticeParameters(lattice_system=lattice_system, grid_size=grid_size)


@pytest.mark.parametrize("lattice_system", LATTICE_SYSTEMS)
@pytest.mark.parametrize("grid_size", [17, 42, 69, 2137])
def test__environment__initializes_properly_with_arbitrary_grid_size(
    env, lattice_system, grid_size
):
    LatticeParameters(lattice_system=lattice_system, grid_size=grid_size)


@pytest.mark.parametrize(
    "lattice_system, exp_state",
    [
        (CUBIC, [0, 0, 0, 30, 30, 30]),
        (HEXAGONAL, [0, 0, 0, 30, 30, 46]),
        (MONOCLINIC, [0, 0, 0, 30, 0, 30]),
        (ORTHORHOMBIC, [0, 0, 0, 30, 30, 30]),
        (RHOMBOHEDRAL, [0, 0, 0, 0, 0, 0]),
        (TETRAGONAL, [0, 0, 0, 30, 30, 30]),
        (TRICLINIC, [0, 0, 0, 0, 0, 0]),
    ],
)
def test__environment__has_expected_initial_state(env, lattice_system, exp_state):
    assert env.state == exp_state


@pytest.mark.parametrize(
    "lattice_system, exp_angles",
    [
        (CUBIC, [90, 90, 90]),
        (HEXAGONAL, [90, 90, 120]),
        (MONOCLINIC, [90, None, 90]),
        (ORTHORHOMBIC, [90, 90, 90]),
        (RHOMBOHEDRAL, [None, None, None]),
        (TETRAGONAL, [90, 90, 90]),
        (TRICLINIC, [None, None, None]),
    ],
)
def test__environment__has_expected_initial_values(env, lattice_system, exp_angles):
    exp_lengths = [env.min_length for _ in range(3)]
    exp_angles = [angle if angle is not None else env.min_angle for angle in exp_angles]

    obs_lengths = [env.cell2length[x] for x in env.state[:3]]
    obs_angles = [env.cell2angle[x] for x in env.state[3:]]

    assert obs_lengths == exp_lengths
    assert obs_angles == exp_angles


@pytest.mark.parametrize("lattice_system", LATTICE_SYSTEMS)
@pytest.mark.parametrize("max_increment", [1, 3, 5])
def test__get_action_space__returns_correct_number_of_actions(
    lattice_system, max_increment
):
    environment = LatticeParameters(
        lattice_system=lattice_system, max_increment=max_increment
    )
    exp_n_actions = 9 * max_increment + 1

    assert len(environment.get_action_space()) == exp_n_actions


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


@pytest.mark.parametrize(
    "lattice_system, state, exp_mask",
    [
        (TRICLINIC, [0, 0, 0, 0, 0, 0], [False] * 9 + [True]),
        (TRICLINIC, [0, 1, 2, 0, 1, 2], [False] * 10),
        (
            CUBIC,
            [0, 0, 0, 30, 30, 30],
            [True, True, True, True, False, True, True, True, True, False],
        ),
        (
            CUBIC,
            [10, 10, 10, 30, 30, 30],
            [True, True, True, True, False, True, True, True, True, False],
        ),
        (
            CUBIC,
            [59, 59, 59, 30, 30, 30],
            [True, True, True, True, False, True, True, True, True, False],
        ),
        (
            MONOCLINIC,
            [0, 0, 0, 30, 0, 30],
            [False, False, False, False, False, True, False, True, True, True],
        ),
        (
            MONOCLINIC,
            [10, 10, 10, 30, 20, 30],
            [False, False, False, False, False, True, False, True, True, True],
        ),
        (
            MONOCLINIC,
            [10, 15, 20, 30, 20, 30],
            [False, False, False, False, False, True, False, True, True, False],
        ),
    ],
)
def test__get_mask_invalid_actions_forward__returns_expected_mask(
    env, lattice_system, state, exp_mask
):
    assert env.get_mask_invalid_actions_forward(state) == exp_mask


@pytest.mark.parametrize("lattice_system", LATTICE_SYSTEMS)
def test__get_parents__returns_no_parents_in_initial_state(env, lattice_system):
    parents, actions = env.get_parents()
    assert len(parents) == 0
    assert len(actions) == 0


@pytest.mark.parametrize("lattice_system", LATTICE_SYSTEMS)
def test__get_parents__returns_parents_after_step(env, lattice_system):
    env.step((1, 1, 1, 0, 0, 0))
    parents, actions = env.get_parents()
    assert len(parents) != 0
    assert len(actions) != 0


@pytest.mark.parametrize("lattice_system", LATTICE_SYSTEMS)
@pytest.mark.parametrize(
    "actions",
    [
        [],
        [(1, 1, 1, 0, 0, 0), (0, 0, 0, 1, 1, 1)],
        [(1, 1, 1, 0, 0, 0)],
        [(1, 1, 1, 0, 0, 0), (1, 1, 1, 0, 0, 0), (1, 1, 1, 0, 0, 0)],
    ],
)
def test__get_parents__returns_same_number_of_parents_and_actions(
    env, lattice_system, actions
):
    for action in actions:
        env.step(action=action)
    parents, actions = env.get_parents()
    assert len(parents) == len(actions)


@pytest.mark.parametrize(
    "lattice_system, actions, exp_state",
    [
        (
            TRICLINIC,
            [
                (1, 0, 0, 0, 0, 0),
                (0, 1, 0, 0, 0, 0),
                (0, 0, 1, 0, 0, 0),
                (0, 0, 0, 1, 0, 0),
                (0, 0, 0, 0, 1, 0),
            ],
            [1, 1, 1, 1, 1, 0],
        ),
        (
            TRICLINIC,
            [(1, 1, 1, 0, 0, 0)],
            [1, 1, 1, 0, 0, 0],
        ),
        (
            TRICLINIC,
            [
                (0, 1, 0, 0, 0, 0),
                (0, 1, 0, 0, 0, 0),
                (1, 0, 0, 0, 0, 0),
                (0, 1, 0, 0, 0, 0),
                (0, 0, 0, 1, 1, 1),
                (1, 1, 1, 0, 0, 0),
            ],
            [2, 4, 1, 1, 1, 1],
        ),
        (
            CUBIC,
            [
                (1, 0, 0, 0, 0, 0),
                (0, 1, 0, 0, 0, 0),
                (0, 0, 1, 0, 0, 0),
                (0, 0, 0, 1, 0, 0),
                (0, 0, 0, 0, 1, 0),
            ],
            [0, 0, 0, 30, 30, 30],
        ),
        (
            CUBIC,
            [
                (1, 0, 0, 0, 0, 0),
                (0, 1, 0, 0, 0, 0),
                (0, 0, 1, 0, 0, 0),
                (0, 0, 0, 1, 0, 0),
                (1, 1, 1, 0, 0, 0),
            ],
            [1, 1, 1, 30, 30, 30],
        ),
        (
            CUBIC,
            [
                (1, 1, 1, 0, 0, 0),
                (1, 1, 1, 0, 0, 0),
                (1, 1, 1, 0, 0, 0),
                (0, 0, 0, 1, 1, 1),
            ],
            [3, 3, 3, 30, 30, 30],
        ),
    ],
)
def test__step__changes_state_as_expected(env, lattice_system, actions, exp_state):
    for action in actions:
        env.step(action=action)

    assert env.state == exp_state


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
            [61, 61, 61, 61, 61, 61],
            [5.0, 5.0, 5.0, 150.0, 150.0, 150.0],
        ),
    ],
)
def test__state2proxy__returns_expected_tensor(env, lattice_system, state, exp_tensor):
    assert torch.equal(env.state2proxy(state)[0], torch.Tensor(exp_tensor))


@pytest.mark.parametrize("lattice_system", [TRICLINIC])
def test__reset(env, lattice_system):
    env.step((1, 1, 1, 0, 0, 0))
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


class TestLattice(common.BaseTestsDiscrete):
    @pytest.fixture(autouse=True)
    def setup(self, triclinic_env):
        self.env = triclinic_env
        self.repeats = {
            "test__reset__state_is_source": 10,
        }
        self.n_states = {}  # TODO: Populate.
