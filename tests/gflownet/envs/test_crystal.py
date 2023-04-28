import pytest
import torch
from torch import Tensor

import common
from gflownet.envs.crystals.crystal import Crystal, Stage
from gflownet.envs.crystals.lattice_parameters import TRICLINIC


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
            Tensor(
                [1.0, 1.0, 1.0, 1.0, 3.0, 1.4444, 1.8889, 2.3333, 90.0, 96.6667, 110.0]
            ),
        ],
        [
            (4, 9, 0, 3, 0, 0, 105, 5, 3, 1, 0, 0, 9),
            Tensor(
                [4.0, 9.0, 0.0, 3.0, 105.0, 3.2222, 2.3333, 1.4444, 30.0, 30.0, 150.0]
            ),
        ],
    ],
)
def test__state2oracle__returns_expected_value(env, state, expected):
    assert torch.allclose(env.state2oracle(state), expected, atol=1e-4)


@pytest.mark.parametrize(
    "state, expected",
    [
        [
            (1, 1, 1, 1, 1, 2, 3, 1, 2, 3, 4, 5, 6),
            Tensor(
                [1.0, 1.0, 1.0, 1.0, 3.0, 1.4444, 1.8889, 2.3333, 90.0, 96.6667, 110.0]
            ),
        ],
        [
            (4, 9, 0, 3, 0, 0, 105, 5, 3, 1, 0, 0, 9),
            Tensor(
                [4.0, 9.0, 0.0, 3.0, 105.0, 3.2222, 2.3333, 1.4444, 30.0, 30.0, 150.0]
            ),
        ],
    ],
)
def test__state2proxy__returns_expected_value(env, state, expected):
    assert torch.allclose(env.state2proxy(state), expected, atol=1e-4)


@pytest.mark.parametrize(
    "batch, expected",
    [
        [
            [
                (1, 1, 1, 1, 1, 2, 3, 1, 2, 3, 4, 5, 6),
                (4, 9, 0, 3, 0, 0, 105, 5, 3, 1, 0, 0, 9),
            ],
            Tensor(
                [
                    [
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        3.0,
                        1.4444,
                        1.8889,
                        2.3333,
                        90.0,
                        96.6667,
                        110.0,
                    ],
                    [
                        4.0,
                        9.0,
                        0.0,
                        3.0,
                        105.0,
                        3.2222,
                        2.3333,
                        1.4444,
                        30.0,
                        30.0,
                        150.0,
                    ],
                ]
            ),
        ],
    ],
)
def test__statebatch2proxy__returns_expected_value(env, batch, expected):
    assert torch.allclose(env.statebatch2proxy(batch), expected, atol=1e-4)


@pytest.mark.parametrize("action", [(1, 1, -2, -2, -2, -2), (3, 4, -2, -2, -2, -2)])
def test__step__single_action_works(env, action):
    env.step(action)

    assert env.state != env.source


@pytest.mark.parametrize(
    "actions, exp_result, exp_stage, last_action_valid",
    [
        [
            [(1, 1, -2, -2, -2, -2), (3, 4, -2, -2, -2, -2)],
            [1, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            Stage.COMPOSITION,
            True,
        ],
        [
            [(2, 225, 3, -3, -3, -3)],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            Stage.COMPOSITION,
            False,
        ],
        [
            [(1, 1, -2, -2, -2, -2), (3, 4, -2, -2, -2, -2), (-1, -1, -2, -2, -2, -2)],
            [1, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            Stage.SPACE_GROUP,
            True,
        ],
        [
            [
                (1, 1, -2, -2, -2, -2),
                (3, 4, -2, -2, -2, -2),
                (-1, -1, -2, -2, -2, -2),
                (2, 105, 0, -3, -3, -3),
            ],
            [1, 0, 4, 0, 4, 3, 105, 0, 0, 0, 0, 0, 0],
            Stage.SPACE_GROUP,
            True,
        ],
        [
            [
                (1, 1, -2, -2, -2, -2),
                (3, 4, -2, -2, -2, -2),
                (-1, -1, -2, -2, -2, -2),
                (2, 105, 0, -3, -3, -3),
                (2, 105, 0, -3, -3, -3),
            ],
            [1, 0, 4, 0, 4, 3, 105, 0, 0, 0, 0, 0, 0],
            Stage.SPACE_GROUP,
            False,
        ],
        [
            [
                (1, 1, -2, -2, -2, -2),
                (3, 4, -2, -2, -2, -2),
                (-1, -1, -2, -2, -2, -2),
                (2, 105, 0, -3, -3, -3),
                (-1, -1, -1, -3, -3, -3),
            ],
            [1, 0, 4, 0, 4, 3, 105, 0, 0, 0, 4, 4, 4],
            Stage.LATTICE_PARAMETERS,
            True,
        ],
        [
            [
                (1, 1, -2, -2, -2, -2),
                (3, 4, -2, -2, -2, -2),
                (-1, -1, -2, -2, -2, -2),
                (2, 105, 0, -3, -3, -3),
                (-1, -1, -1, -3, -3, -3),
                (1, 0, 0, 0, 0, 0),
            ],
            [1, 0, 4, 0, 4, 3, 105, 0, 0, 0, 4, 4, 4],
            Stage.LATTICE_PARAMETERS,
            False,
        ],
        [
            [
                (1, 1, -2, -2, -2, -2),
                (3, 4, -2, -2, -2, -2),
                (-1, -1, -2, -2, -2, -2),
                (2, 105, 0, -3, -3, -3),
                (-1, -1, -1, -3, -3, -3),
                (1, 1, 1, 0, 0, 0),
            ],
            [1, 0, 4, 0, 4, 3, 105, 1, 1, 1, 4, 4, 4],
            Stage.LATTICE_PARAMETERS,
            True,
        ],
        [
            [
                (1, 1, -2, -2, -2, -2),
                (3, 4, -2, -2, -2, -2),
                (-1, -1, -2, -2, -2, -2),
                (2, 105, 0, -3, -3, -3),
                (-1, -1, -1, -3, -3, -3),
                (1, 1, 1, 0, 0, 0),
                (0, 0, 0, 0, 0, 0),
            ],
            [1, 0, 4, 0, 4, 3, 105, 1, 1, 1, 4, 4, 4],
            Stage.LATTICE_PARAMETERS,
            False,
        ],
        [
            [
                (1, 1, -2, -2, -2, -2),
                (3, 4, -2, -2, -2, -2),
                (-1, -1, -2, -2, -2, -2),
                (2, 105, 0, -3, -3, -3),
                (-1, -1, -1, -3, -3, -3),
                (1, 1, 1, 0, 0, 0),
                (1, 1, 0, 0, 0, 0),
                (0, 0, 0, 0, 0, 0),
            ],
            [1, 0, 4, 0, 4, 3, 105, 2, 2, 1, 4, 4, 4],
            Stage.LATTICE_PARAMETERS,
            True,
        ],
    ],
)
def test__step__action_sequence_has_expected_result(
    env, actions, exp_result, exp_stage, last_action_valid
):
    for action in actions:
        _, _, valid = env.step(action)

    assert env.state == exp_result
    assert env.stage == exp_stage
    assert valid == last_action_valid


def test__get_parents__returns_no_parents_in_initial_state(env):
    return common.test__get_parents__returns_no_parents_in_initial_state(env)


@pytest.mark.parametrize(
    "actions",
    [
        [(1, 1, -2, -2, -2, -2), (3, 4, -2, -2, -2, -2)],
        [
            (1, 1, -2, -2, -2, -2),
            (3, 4, -2, -2, -2, -2),
            (-1, -1, -2, -2, -2, -2),
            (2, 105, 0, -3, -3, -3),
            (-1, -1, -1, -3, -3, -3),
            (1, 1, 1, 0, 0, 0),
            (1, 1, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0),
        ],
    ],
)
def test__get_parents__contains_previous_action_after_a_step(env, actions):
    for action in actions:
        env.step(action)
        parents, parent_actions = env.get_parents()
        assert action in parent_actions


@pytest.mark.parametrize(
    "actions",
    [
        [
            (1, 1, -2, -2, -2, -2),
            (3, 4, -2, -2, -2, -2),
            (-1, -1, -2, -2, -2, -2),
            (2, 105, 0, -3, -3, -3),
            (-1, -1, -1, -3, -3, -3),
            (1, 1, 1, 0, 0, 0),
            (1, 1, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0),
        ]
    ],
)
def test__reset(env, actions):
    for action in actions:
        env.step(action)

    assert env.state != env.source
    for subenv in [env.composition, env.space_group, env.lattice_parameters]:
        assert subenv.state != subenv.source
    assert env.lattice_parameters.lattice_system != TRICLINIC

    env.reset()

    assert env.state == env.source
    for subenv in [env.composition, env.space_group, env.lattice_parameters]:
        assert subenv.state == subenv.source
    assert env.lattice_parameters.lattice_system == TRICLINIC


@pytest.mark.parametrize(
    "actions, exp_stage",
    [
        [
            [],
            Stage.COMPOSITION,
        ],
        [
            [(1, 1, -2, -2, -2, -2), (3, 4, -2, -2, -2, -2), (-1, -1, -2, -2, -2, -2)],
            Stage.SPACE_GROUP,
        ],
        [
            [
                (1, 1, -2, -2, -2, -2),
                (3, 4, -2, -2, -2, -2),
                (-1, -1, -2, -2, -2, -2),
                (2, 105, 0, -3, -3, -3),
                (-1, -1, -1, -3, -3, -3),
            ],
            Stage.LATTICE_PARAMETERS,
        ],
    ],
)
def test__get_mask_invalid_actions_forward__masks_all_actions_from_different_stages(
    env, actions, exp_stage
):
    for action in actions:
        env.step(action)

    assert env.stage == exp_stage

    mask = env.get_mask_invalid_actions_forward()

    if env.stage == Stage.COMPOSITION:
        assert not all(mask[: len(env.composition.action_space)])
        assert all(mask[len(env.composition.action_space) :])
    if env.stage == Stage.SPACE_GROUP:
        assert not all(
            mask[
                len(env.composition.action_space) : len(env.composition.action_space)
                + len(env.space_group.action_space)
            ]
        )
        assert all(mask[: len(env.composition.action_space)])
        assert all(
            mask[
                len(env.composition.action_space) + len(env.space_group.action_space) :
            ]
        )
    if env.stage == Stage.LATTICE_PARAMETERS:
        assert not all(
            mask[
                len(env.composition.action_space) + len(env.space_group.action_space) :
            ]
        )
        assert all(
            mask[
                : len(env.composition.action_space) + len(env.space_group.action_space)
            ]
        )
