import common
import numpy as np
import pymatgen.symmetry.groups as pmgg
import pytest
import torch
from pyxtal.symmetry import Group

from gflownet.envs.crystals.spacegroup import SpaceGroup

N_ATOMS = [3, 7, 9]
SG_SUBSET = [1, 17, 39, 123, 230]


@pytest.fixture
def env():
    return SpaceGroup()


@pytest.fixture
def env_with_composition():
    return SpaceGroup(n_atoms=N_ATOMS)


@pytest.fixture
def env_with_restricted_spacegroups():
    return SpaceGroup(space_groups_subset=SG_SUBSET)


def test__environment__initializes_properly():
    env = SpaceGroup()
    assert env.source == [0] * 3
    assert env.state == [0] * 3


def test__environment__space_groups_subset__initializes_properly():
    def count_distinct(my_dict, sub_key):
        all_elements = []
        for sub_dict in my_dict.values():
            all_elements.extend(sub_dict[sub_key])

        distinct_elements = set(all_elements)
        return len(distinct_elements)

    env = SpaceGroup(space_groups_subset=[1, 2])
    nb_spacegroups = 2
    nb_cls = 1
    nb_ps = 2
    assert env.source == [0] * 3
    assert env.state == [0] * 3
    assert len(env.space_groups) == nb_spacegroups
    assert len(env.crystal_lattice_systems) == nb_cls
    assert len(env.point_symmetries) == nb_ps
    assert count_distinct(env.crystal_lattice_systems, "space_groups") == nb_spacegroups
    assert count_distinct(env.crystal_lattice_systems, "point_symmetries") == nb_ps
    assert count_distinct(env.point_symmetries, "space_groups") == nb_spacegroups
    assert count_distinct(env.point_symmetries, "crystal_lattice_systems") == nb_cls

    env = SpaceGroup(space_groups_subset=range(1, 15 + 1))
    nb_spacegroups = 15
    nb_cls = 2
    nb_ps = 3
    assert env.source == [0] * 3
    assert env.state == [0] * 3
    assert len(env.space_groups) == nb_spacegroups
    assert len(env.crystal_lattice_systems) == nb_cls
    assert len(env.point_symmetries) == nb_ps
    assert count_distinct(env.crystal_lattice_systems, "space_groups") == nb_spacegroups
    assert count_distinct(env.crystal_lattice_systems, "point_symmetries") == nb_ps
    assert count_distinct(env.point_symmetries, "space_groups") == nb_spacegroups
    assert count_distinct(env.point_symmetries, "crystal_lattice_systems") == nb_cls

    env = SpaceGroup(space_groups_subset=SG_SUBSET)
    nb_spacegroups = len(SG_SUBSET)
    nb_cls = 4
    nb_ps = 4
    assert env.source == [0] * 3
    assert env.state == [0] * 3
    assert len(env.space_groups) == nb_spacegroups
    assert len(env.crystal_lattice_systems) == nb_cls
    assert len(env.point_symmetries) == nb_ps
    assert count_distinct(env.crystal_lattice_systems, "space_groups") == nb_spacegroups
    assert count_distinct(env.crystal_lattice_systems, "point_symmetries") == nb_ps
    assert count_distinct(env.point_symmetries, "space_groups") == nb_spacegroups
    assert count_distinct(env.point_symmetries, "crystal_lattice_systems") == nb_cls


def test__environment__action_space_has_eos():
    env = SpaceGroup()
    assert env.eos in env.action_space


@pytest.mark.parametrize(
    "action, expected",
    [
        (
            (0, 1, 0),
            True,
        ),
        (
            (0, 1, 1),
            False,
        ),
        (
            (0, 1, 2),
            True,
        ),
        (
            (0, 1, 3),
            False,
        ),
        (
            (1, 1, 0),
            True,
        ),
        (
            (1, 1, 1),
            True,
        ),
        (
            (1, 1, 2),
            False,
        ),
        (
            (1, 1, 3),
            False,
        ),
        (
            (2, 1, 0),
            True,
        ),
        (
            (2, 1, 1),
            True,
        ),
        (
            (2, 1, 2),
            True,
        ),
        (
            (2, 1, 3),
            True,
        ),
    ],
)
def test__action_space__contains_expected(env, action, expected):
    assert (action in env.action_space) == expected


@pytest.mark.parametrize(
    "action, expected",
    [
        (
            (2, 1, 0),
            True,
        ),
        (
            (2, 17, 0),
            True,
        ),
        (
            (2, 39, 0),
            True,
        ),
        (
            (2, 123, 0),
            True,
        ),
        (
            (2, 230, 0),
            True,
        ),
        (
            (2, 2, 0),
            False,
        ),
    ],
)
def test__action_space__env_with_restricted_spacegroups__contains_expected(
    env_with_restricted_spacegroups, action, expected
):
    env = env_with_restricted_spacegroups
    assert (action in env.action_space) == expected


@pytest.mark.parametrize(
    "state, action, expected",
    [
        (
            [0, 0, 0],
            (0, 1, 0),
            False,
        ),
        (
            [0, 0, 0],
            (0, 1, 2),
            True,
        ),
        (
            [0, 0, 0],
            (1, 1, 0),
            False,
        ),
        (
            [0, 0, 0],
            (2, 1, 0),
            False,
        ),
        (
            [0, 0, 1],
            (2, 1, 0),
            True,
        ),
        (
            [0, 0, 1],
            (0, 1, 0),
            True,
        ),
        (
            [0, 0, 1],
            (1, 1, 0),
            True,
        ),
        (
            [1, 1, 0],
            (2, 1, 3),
            False,
        ),
    ],
)
def test__get_mask_invalid_actions_forward__masks_expected_action(
    env, state, action, expected
):
    assert action in env.action_space
    mask = env.get_mask_invalid_actions_forward(state, False)
    assert mask[env.action_space.index(action)] == expected, print(
        state, action, expected
    )


@pytest.mark.parametrize(
    "state, parents_expected, parents_a_expected",
    [
        (
            [7, 3, 184],
            [[0, 0, 0], [7, 3, 0], [7, 0, 0], [0, 3, 0]],
            [(2, 184, 0), (2, 184, 3), (2, 184, 1), (2, 184, 2)],
        ),
        (
            [1, 0, 0],
            [[0, 0, 0]],
            [(0, 1, 0)],
        ),
        (
            [0, 1, 0],
            [[0, 0, 0]],
            [(1, 1, 0)],
        ),
        (
            [5, 4, 0],
            [[5, 0, 0], [0, 4, 0]],
            [(1, 4, 1), (0, 5, 2)],
        ),
        (
            [8, 2, 0],
            [[8, 0, 0], [0, 2, 0]],
            [(1, 2, 1), (0, 8, 2)],
        ),
    ],
)
def test__get_parents__returns_expected(
    env, state, parents_expected, parents_a_expected
):
    parents, parents_a = env.get_parents(state)
    parents = [tuple(p) for p in parents]
    parents_expected = [tuple(p) for p in parents_expected]
    assert set(parents) == set(parents_expected)
    assert set(parents_a) == set(parents_a_expected)


@pytest.mark.parametrize(
    "state",
    [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ],
)
def test__state2readable2state(env, state):
    assert all(
        [
            el1 == el2
            for el1, el2 in zip(env.readable2state(env.state2readable(state)), state)
        ]
    )


def test__env_with_composition__compatibility_dict_as_in_pyxtal(env_with_composition):
    for (
        sg,
        is_compatible,
    ) in env_with_composition.n_atoms_compatibility_dict.items():
        sg_pyxtal = Group(sg)
        assert sg_pyxtal.check_compatible(N_ATOMS)[0] == is_compatible


def test__get_mask_invalid_actions_forward__incompatible_sg_are_invalid(
    env_with_composition,
):
    """
    For all states with crystal-lattice system and point symmetry but not space group
    set, check that incompatible space groups according to pyxtal correspond to invalid
    actions.

    Note that this test takes non-negligible time. You may skip it for debugging/dev
    with: @pytest.mark.skip(reason="disabled during debugging - reactivate!")
    """
    all_x = env_with_composition.get_all_terminating_states()
    for state in all_x:
        state[env_with_composition.sg_idx] = 0
        env_with_composition.set_state(state=state, done=False)
        mask_f = env_with_composition.get_mask_invalid_actions_forward()
        state_type = env_with_composition.get_state_type(state)
        for sg in env_with_composition.space_groups:
            sg_pyxtal = Group(sg)
            is_compatible = sg_pyxtal.check_compatible(N_ATOMS)[0]
            action = (env_with_composition.sg_idx, sg, state_type)
            if not is_compatible:
                assert mask_f[env_with_composition.action_space.index(action)] is True


def test__states_are_compatible_with_pymatgen(env):
    for idx in env.space_groups:
        env = env.reset()
        env.step((2, idx, 0))
        sg_int = pmgg.sg_symbol_from_int_number(idx)
        sg = pmgg.SpaceGroup(sg_int)
        assert sg.int_number == env.state[env.sg_idx]
        assert sg.crystal_system == env.crystal_system
        assert sg.symbol == env.space_group_symbol
        assert sg.point_group == env.point_group


@pytest.mark.parametrize(
    "n_atoms, cls_idx, ps_idx",
    [
        [[1], 5, 1],
        [[17], 5, 1],
        [[1, 13], 5, 1],
    ],
)
def test__special_cases_composition_compatibility(n_atoms, cls_idx, ps_idx):
    env = SpaceGroup(n_atoms=n_atoms)
    # Crystal lattice system space groups must not compatible with composition
    # constraints
    assert env._is_compatible(cls_idx=cls_idx) is False
    # Setting crystal lattice system should fail
    action_cls_5_from_0 = (0, 5, 0)
    state_new, action, valid = env.step(action_cls_5_from_0)
    assert valid is False
    # Point symmetry space groups must be compatible with composition constraints
    assert env._is_compatible(ps_idx=ps_idx) is True
    # Setting point symmetry should be valid
    action_ps_1_from_0 = (1, 1, 0)
    state_new, action, valid = env.step(action_ps_1_from_0)
    assert valid is True
    # Setting crystal lattice system should still fail
    action_cls_5_from_2 = (0, 5, 2)
    state_new, action, valid = env.step(action_cls_5_from_2)
    assert valid is False


def test__all_common__env(env):
    print("\n\nCommon tests for SpaceGroup without composition restrictions\n")
    return common.test__all_env_common(env)


def test__all_common__env_with_composition(env_with_composition):
    print(
        f"\n\nCommon tests for SpaceGroup with restrictions from composition {N_ATOMS}\n"
    )
    return common.test__all_env_common(env_with_composition)


def test__all_common__env_with_restricted_spacegroups(env_with_restricted_spacegroups):
    print(f"\n\nCommon tests for SpaceGroup with restricted space groups {SG_SUBSET}")
    return common.test__all_env_common(env_with_restricted_spacegroups)
