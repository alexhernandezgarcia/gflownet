import pytest
import torch
import numpy as np

import pymatgen.symmetry.groups as pmgg
from gflownet.envs.spacegroup import SpaceGroup


@pytest.fixture
def env():
    return SpaceGroup()


def test__environment__initializes_properly():
    env = SpaceGroup()
    assert env.source == [0] * 3
    assert env.state == [0] * 3


def test__environment__action_space_has_eos():
    env = SpaceGroup()
    assert (env.eos, 0) in env.action_space


@pytest.mark.parametrize(
    "state, action, expected",
    [
        (
            [0, 0, 0],
            (0, 1),
            False,
        ),
        (
            [1, 0, 0],
            (0, 1),
            True,
        ),
        (
            [0, 0, 0],
            (1, 1),
            False,
        ),
        (
            [0, 1, 0],
            (1, 1),
            True,
        ),
        (
            [0, 0, 0],
            (2, 1),
            False,
        ),
        (
            [0, 0, 1],
            (2, 1),
            True,
        ),
        (
            [0, 0, 1],
            (0, 1),
            True,
        ),
        (
            [0, 0, 1],
            (1, 1),
            True,
        ),
        (
            [1, 1, 0],
            (2, 1),
            False,
        ),
    ],
)
def test__get_mask_invalid_actions_forward__masks_expected_action(env, state, action, expected):
    assert action in env.action_space
    mask = env.get_mask_invalid_actions_forward(state, False)
    assert mask[env.action_space.index(action)] == expected


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


# TODO: make common to all environments
def test__get_parents_step_get_mask__are_compatible(env, n=100):
    for traj in range(n):
        env = env.reset()
        while not env.done:
            mask_invalid = env.get_mask_invalid_actions_forward()
            valid_actions = [a for a, m in zip(env.action_space, mask_invalid) if not m]
            action = tuple(np.random.permutation(valid_actions)[0])
            env.step(action)
            parents, parents_a = env.get_parents()
            assert len(parents) == len(parents_a)
            for p, p_a in zip(parents, parents_a):
                mask = env.get_mask_invalid_actions_forward(p, False)
                assert p_a in env.action_space
                assert mask[env.action_space.index(p_a)] == False


def test__states_are_compatible_with_pymatgen(env):
    for idx in range(env.n_space_groups):
        env = env.reset()
        env.step((2, idx + 1))
        sg_int = pmgg.sg_symbol_from_int_number(idx + 1)
        sg = pmgg.SpaceGroup(sg_int)
        assert sg.int_number == env.state[env.sg_idx]
        assert sg.crystal_system == env.crystal_systems[env.state[env.cs_idx]][0]
        crystal_class_idx = env.space_groups[idx + 1][1]
        point_groups = env.crystal_classes[crystal_class_idx][2]
        assert sg.point_group in point_groups


# TODO: make common to all environments
def test__get_parents__returns_no_parents_in_initial_state(env):
    parents, actions = env.get_parents()
    assert len(parents) == 0
    assert len(actions) == 0
