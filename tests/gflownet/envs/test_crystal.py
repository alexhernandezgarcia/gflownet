import pytest
import torch
from torch import Tensor

from gflownet.envs.crystals.crystal import Crystal, Stage


@pytest.fixture
def env():
    return Crystal(
        composition_kwargs={"elements": 4}, lattice_parameters_kwargs={"grid_size": 10}
    )


def test__environment__initializes_properly(env):
    pass


def test__environment__has_expected_initial_state(env):
    assert (
        env.state == env.source == [0] * (4 + 3 + 6)
    )  # n elements + space groups + lattice parameters


def test__environment__has_expected_action_space(env):
    assert len(env.action_space) == len(env.composition.action_space) + len(
        env.space_group.action_space
    ) + len(env.lattice_parameters.action_space)

    underlying_action_space = (
        env.composition.action_space
        + env.space_group.action_space
        + env.lattice_parameters.action_space
    )

    for action, underlying_action in zip(env.action_space, underlying_action_space):
        assert action[: len(underlying_action)] == underlying_action


def test__pad_depad_action(env):
    for subenv, stage in [
        (env.composition, Stage.COMPOSITION),
        (env.space_group, Stage.SPACE_GROUP),
        (env.lattice_parameters, Stage.LATTICE_PARAMETERS),
    ]:
        for action in subenv.action_space:
            padded = env._pad_action(action, stage)
            assert len(padded) == env.max_action_length
            depadded = env._depad_action(padded, stage)
            assert depadded == action


@pytest.mark.parametrize(
    "state, expected",
    [
        [
            (1, 1, 1, 1, 1, 2, 3, 1, 2, 3, 4, 5, 6),
            [
                1.0,
                4.0,
                0.25,
                0.25,
                0.25,
                0.25,
                3.0,
                1.4444,
                1.8889,
                2.3333,
                90.0,
                96.6667,
                110.0,
            ],
        ]
    ],
)
def test__state2oracle__returns_expected_value(env, state, expected):
    assert torch.allclose(env.state2oracle(state), Tensor(expected), atol=1e-4)
