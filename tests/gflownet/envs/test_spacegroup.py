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
def test__get_mask_invalid_actions_forward__right_masks(env, state, action, expected):
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
