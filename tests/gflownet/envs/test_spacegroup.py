import common
import numpy as np
import pymatgen.symmetry.groups as pmgg
import pytest
import torch

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
    assert env.eos in env.action_space


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
def test__get_mask_invalid_actions_forward__masks_expected_action(
    env, state, action, expected
):
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


@pytest.mark.skip(reason="Takes considerable time")
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


def test__all_env_common(env):
    return common.test__all_env_common(env)
