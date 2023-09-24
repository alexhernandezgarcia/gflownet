import warnings

import common
import numpy as np
import pytest
import torch
from torch import Tensor

from gflownet.envs.crystals.ccrystal import CCrystal, Stage
from gflownet.envs.crystals.clattice_parameters import TRICLINIC
from gflownet.utils.common import tbool, tfloat


@pytest.fixture
def env():
    return CCrystal(
        composition_kwargs={"elements": 4},
        space_group_kwargs={"space_groups_subset": list(range(1, 15 + 1)) + [105]},
    )


@pytest.fixture
def env_with_stoichiometry_sg_check():
    return CCrystal(
        composition_kwargs={"elements": 4},
        do_stoichiometry_sg_check=True,
    )


def test__environment__initializes_properly(env):
    pass


def test__environment__has_expected_initial_state(env):
    """
    The source of the composition and space group environments is all 0s. The source of
    the continuous lattice parameters environment is all -1s.
    """
    assert (
        env.state == env.source == [0] * (1 + 4 + 3) + [-1] * 6
    )  # stage + n elements + space groups + lattice parameters


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
    for stage, subenv in env.subenvs.items():
        for action in subenv.action_space:
            padded = env._pad_action(action, stage)
            assert len(padded) == env.max_action_length
            depadded = env._depad_action(padded, stage)
            assert depadded == action


@pytest.mark.skip(reason="skip until updated")
@pytest.mark.parametrize(
    "state, expected",
    [
        [
            (2, 1, 1, 1, 1, 1, 2, 3, 1, 2, 3, 4, 5, 6),
            Tensor([1.0, 1.0, 1.0, 1.0, 3.0, 1.4, 1.8, 2.2, 78.0, 90.0, 102.0]),
        ],
        [
            (2, 4, 9, 0, 3, 0, 0, 105, 5, 3, 1, 0, 0, 9),
            Tensor([4.0, 9.0, 0.0, 3.0, 105.0, 3.0, 2.2, 1.4, 30.0, 30.0, 138.0]),
        ],
    ],
)
def test__state2oracle__returns_expected_value(env, state, expected):
    assert torch.allclose(env.state2oracle(state), expected, atol=1e-4)


@pytest.mark.skip(reason="skip until updated")
@pytest.mark.parametrize(
    "state, expected",
    [
        [
            (2, 1, 1, 1, 1, 1, 2, 3, 1, 2, 3, 4, 5, 6),
            Tensor([1.0, 1.0, 1.0, 1.0, 3.0, 1.4, 1.8, 2.2, 78.0, 90.0, 102.0]),
        ],
        [
            (2, 4, 9, 0, 3, 0, 0, 105, 5, 3, 1, 0, 0, 9),
            Tensor([4.0, 9.0, 0.0, 3.0, 105.0, 3.0, 2.2, 1.4, 30.0, 30.0, 138.0]),
        ],
    ],
)
def test__state2proxy__returns_expected_value(env, state, expected):
    assert torch.allclose(env.state2proxy(state), expected, atol=1e-4)


@pytest.mark.skip(reason="skip until updated")
@pytest.mark.parametrize(
    "batch, expected",
    [
        [
            [
                (2, 1, 1, 1, 1, 1, 2, 3, 1, 2, 3, 4, 5, 6),
                (2, 4, 9, 0, 3, 0, 0, 105, 5, 3, 1, 0, 0, 9),
            ],
            Tensor(
                [
                    [1.0, 1.0, 1.0, 1.0, 3.0, 1.4, 1.8, 2.2, 78.0, 90.0, 102.0],
                    [4.0, 9.0, 0.0, 3.0, 105.0, 3.0, 2.2, 1.4, 30.0, 30.0, 138.0],
                ]
            ),
        ],
    ],
)
def test__statebatch2proxy__returns_expected_value(env, batch, expected):
    assert torch.allclose(env.statebatch2proxy(batch), expected, atol=1e-4)


@pytest.mark.parametrize(
    "action", [(1, 1, -2, -2, -2, -2, -2), (3, 4, -2, -2, -2, -2, -2)]
)
def test__step__single_action_works(env, action):
    env.step(action)

    assert env.state != env.source


@pytest.mark.parametrize(
    "actions, exp_result, exp_stage, last_action_valid",
    [
        [
            [(1, 1, -2, -2, -2, -2, -2), (3, 4, -2, -2, -2, -2, -2)],
            [0, 1, 0, 4, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1],
            Stage.COMPOSITION,
            True,
        ],
        [
            [(2, 105, 3, -3, -3, -3, -3)],
            [0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1],
            Stage.COMPOSITION,
            False,
        ],
        [
            [
                (1, 1, -2, -2, -2, -2, -2),
                (3, 4, -2, -2, -2, -2, -2),
                (-1, -1, -2, -2, -2, -2, -2),
            ],
            [1, 1, 0, 4, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1],
            Stage.SPACE_GROUP,
            True,
        ],
        [
            [
                (1, 1, -2, -2, -2, -2, -2),
                (3, 4, -2, -2, -2, -2, -2),
                (-1, -1, -2, -2, -2, -2, -2),
                (2, 105, 0, -3, -3, -3, -3),
            ],
            [1, 1, 0, 4, 0, 4, 3, 105, -1, -1, -1, -1, -1, -1],
            Stage.SPACE_GROUP,
            True,
        ],
        [
            [
                (1, 1, -2, -2, -2, -2, -2),
                (3, 4, -2, -2, -2, -2, -2),
                (-1, -1, -2, -2, -2, -2, -2),
                (2, 105, 0, -3, -3, -3, -3),
                (2, 105, 0, -3, -3, -3, -3),
            ],
            [1, 1, 0, 4, 0, 4, 3, 105, -1, -1, -1, -1, -1, -1],
            Stage.SPACE_GROUP,
            False,
        ],
        [
            [
                (1, 1, -2, -2, -2, -2, -2),
                (3, 4, -2, -2, -2, -2, -2),
                (-1, -1, -2, -2, -2, -2, -2),
                (2, 105, 0, -3, -3, -3, -3),
                (-1, -1, -1, -3, -3, -3, -3),
            ],
            [2, 1, 0, 4, 0, 4, 3, 105, -1, -1, -1, -1, -1, -1],
            Stage.LATTICE_PARAMETERS,
            True,
        ],
        [
            [
                (1, 1, -2, -2, -2, -2, -2),
                (3, 4, -2, -2, -2, -2, -2),
                (-1, -1, -2, -2, -2, -2, -2),
                (2, 105, 0, -3, -3, -3, -3),
                (-1, -1, -1, -3, -3, -3, -3),
                (1.5, 0, 0, 0, 0, 0, 0),
            ],
            [2, 1, 0, 4, 0, 4, 3, 105, -1, -1, -1, -1, -1, -1],
            Stage.LATTICE_PARAMETERS,
            False,
        ],
        [
            [
                (1, 1, -2, -2, -2, -2, -2),
                (3, 4, -2, -2, -2, -2, -2),
                (-1, -1, -2, -2, -2, -2, -2),
                (2, 105, 0, -3, -3, -3, -3),
                (-1, -1, -1, -3, -3, -3, -3),
                (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1),
            ],
            [2, 1, 0, 4, 0, 4, 3, 105, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            Stage.LATTICE_PARAMETERS,
            True,
        ],
        [
            [
                (1, 1, -2, -2, -2, -2, -2),
                (3, 4, -2, -2, -2, -2, -2),
                (-1, -1, -2, -2, -2, -2, -2),
                (2, 105, 0, -3, -3, -3, -3),
                (-1, -1, -1, -3, -3, -3, -3),
                (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1),
                (0.6, 0.5, 0.4, 0.3, 0.2, 0.6, 0),
            ],
            [2, 1, 0, 4, 0, 4, 3, 105, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            Stage.LATTICE_PARAMETERS,
            False,
        ],
        [
            [
                (1, 1, -2, -2, -2, -2, -2),
                (3, 4, -2, -2, -2, -2, -2),
                (-1, -1, -2, -2, -2, -2, -2),
                (2, 105, 0, -3, -3, -3, -3),
                (-1, -1, -1, -3, -3, -3, -3),
                (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1),
                (0.66, 0.55, 0.44, 0.33, 0.22, 0.11, 0),
            ],
            [2, 1, 0, 4, 0, 4, 3, 105, 0.76, 0.75, 0.74, 0.73, 0.72, 0.71],
            Stage.LATTICE_PARAMETERS,
            True,
        ],
        [
            [
                (1, 1, -2, -2, -2, -2, -2),
                (3, 4, -2, -2, -2, -2, -2),
                (-1, -1, -2, -2, -2, -2, -2),
                (2, 105, 0, -3, -3, -3, -3),
                (-1, -1, -1, -3, -3, -3, -3),
                (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1),
                (0.66, 0.55, 0.44, 0.33, 0.22, 0.11, 0),
                (np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf),
            ],
            [2, 1, 0, 4, 0, 4, 3, 105, 0.76, 0.75, 0.74, 0.73, 0.72, 0.71],
            Stage.LATTICE_PARAMETERS,
            True,
        ],
    ],
)
def test__step__action_sequence_has_expected_result(
    env, actions, exp_result, exp_stage, last_action_valid
):
    for action in actions:
        warnings.filterwarnings("ignore")
        _, _, valid = env.step(action)

    assert env.state == exp_result
    assert env._get_stage() == exp_stage
    assert valid == last_action_valid


# TODO: Remove if get_parents is removed
@pytest.mark.parametrize(
    "actions",
    [
        [(1, 1, -2, -2, -2, -2, -2), (3, 4, -2, -2, -2, -2, -2)],
        [
            (1, 1, -2, -2, -2, -2, -2),
            (3, 4, -2, -2, -2, -2, -2),
            (-1, -1, -2, -2, -2, -2, -2),
            (2, 105, 0, -3, -3, -3, -3),
            (-1, -1, -1, -3, -3, -3, -3),
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
            (1, 1, -2, -2, -2, -2, -2),
            (3, 4, -2, -2, -2, -2, -2),
            (-1, -1, -2, -2, -2, -2, -2),
            (2, 105, 0, -3, -3, -3, -3),
            (-1, -1, -1, -3, -3, -3, -3),
            (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1),
            (0.66, 0.55, 0.44, 0.33, 0.22, 0.11, 0),
            (np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf),
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


@pytest.mark.skip(reason="skip while developping other tests")
@pytest.mark.parametrize(
    "actions, exp_stage",
    [
        [
            [],
            Stage.COMPOSITION,
        ],
        [
            [
                (1, 1, -2, -2, -2, -2, -2),
                (3, 4, -2, -2, -2, -2, -2),
                (-1, -1, -2, -2, -2, -2, -2),
            ],
            Stage.SPACE_GROUP,
        ],
        [
            [
                (1, 1, -2, -2, -2, -2, -2),
                (3, 4, -2, -2, -2, -2, -2),
                (-1, -1, -2, -2, -2, -2, -2),
                (2, 105, 0, -3, -3, -3, -3),
                (-1, -1, -1, -3, -3, -3, -3),
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

    assert env._get_stage() == exp_stage

    mask = env.get_mask_invalid_actions_forward()

    if env._get_stage() == Stage.COMPOSITION:
        assert not all(mask[: len(env.composition.action_space)])
        assert all(mask[len(env.composition.action_space) :])
    if env._get_stage() == Stage.SPACE_GROUP:
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
    if env._get_stage() == Stage.LATTICE_PARAMETERS:
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


def test__get_policy_outputs__is_the_concatenation_of_subenvs(env):
    policy_output_composition = env.composition.get_policy_output(
        env.composition.fixed_distr_params
    )
    policy_output_space_group = env.space_group.get_policy_output(
        env.space_group.fixed_distr_params
    )
    policy_output_lattice_parameters = env.lattice_parameters.get_policy_output(
        env.lattice_parameters.fixed_distr_params
    )
    policy_output_cat = torch.cat(
        (
            policy_output_composition,
            policy_output_space_group,
            policy_output_lattice_parameters,
        )
    )
    policy_output = env.get_policy_output(env.fixed_distr_params)
    assert torch.all(torch.eq(policy_output_cat, policy_output))


def test___get_policy_outputs_of_subenv__returns_correct_output(env):
    n_states = 5
    policy_output_composition = torch.tile(
        env.composition.get_policy_output(env.composition.fixed_distr_params),
        dims=(n_states, 1),
    )
    policy_output_space_group = torch.tile(
        env.space_group.get_policy_output(env.space_group.fixed_distr_params),
        dims=(n_states, 1),
    )
    policy_output_lattice_parameters = torch.tile(
        env.lattice_parameters.get_policy_output(
            env.lattice_parameters.fixed_distr_params
        ),
        dims=(n_states, 1),
    )
    policy_outputs = torch.tile(
        env.get_policy_output(env.fixed_distr_params), dims=(n_states, 1)
    )
    assert torch.all(
        torch.eq(
            env._get_policy_outputs_of_subenv(policy_outputs, Stage.COMPOSITION),
            policy_output_composition,
        )
    )
    assert torch.all(
        torch.eq(
            env._get_policy_outputs_of_subenv(policy_outputs, Stage.SPACE_GROUP),
            policy_output_space_group,
        )
    )
    assert torch.all(
        torch.eq(
            env._get_policy_outputs_of_subenv(policy_outputs, Stage.LATTICE_PARAMETERS),
            policy_output_lattice_parameters,
        )
    )


@pytest.mark.parametrize(
    "state, state_composition, state_space_group, state_lattice_parameters",
    [
        [
            [0, 1, 0, 4, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1],
            [1, 0, 4, 0],
            [0, 0, 0],
            [-1, -1, -1, -1, -1, -1],
        ],
        [
            [0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1],
            [0, 0, 0, 0],
            [0, 0, 0],
            [-1, -1, -1, -1, -1, -1],
        ],
        [
            [1, 1, 0, 4, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1],
            [1, 0, 4, 0],
            [0, 0, 0],
            [-1, -1, -1, -1, -1, -1],
        ],
        [
            [1, 1, 0, 4, 0, 4, 3, 105, -1, -1, -1, -1, -1, -1],
            [1, 0, 4, 0],
            [4, 3, 105],
            [-1, -1, -1, -1, -1, -1],
        ],
        [
            [1, 1, 0, 4, 0, 4, 3, 105, -1, -1, -1, -1, -1, -1],
            [1, 0, 4, 0],
            [4, 3, 105],
            [-1, -1, -1, -1, -1, -1],
        ],
        [
            [2, 1, 0, 4, 0, 4, 3, 105, -1, -1, -1, -1, -1, -1],
            [1, 0, 4, 0],
            [4, 3, 105],
            [-1, -1, -1, -1, -1, -1],
        ],
        [
            [2, 1, 0, 4, 0, 4, 3, 105, -1, -1, -1, -1, -1, -1],
            [1, 0, 4, 0],
            [4, 3, 105],
            [-1, -1, -1, -1, -1, -1],
        ],
        [
            [2, 1, 0, 4, 0, 4, 3, 105, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            [1, 0, 4, 0],
            [4, 3, 105],
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        ],
        [
            [2, 1, 0, 4, 0, 4, 3, 105, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            [1, 0, 4, 0],
            [4, 3, 105],
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        ],
        [
            [2, 1, 0, 4, 0, 4, 3, 105, 0.76, 0.75, 0.74, 0.73, 0.72, 0.71],
            [1, 0, 4, 0],
            [4, 3, 105],
            [0.76, 0.75, 0.74, 0.73, 0.72, 0.71],
        ],
        [
            [2, 1, 0, 4, 0, 4, 3, 105, 0.76, 0.75, 0.74, 0.73, 0.72, 0.71],
            [1, 0, 4, 0],
            [4, 3, 105],
            [0.76, 0.75, 0.74, 0.73, 0.72, 0.71],
        ],
    ],
)
def test__state_of_subenv__returns_expected(
    env, state, state_composition, state_space_group, state_lattice_parameters
):
    for stage in env.subenvs:
        state_subenv = env._get_state_of_subenv(state, stage)
        if stage == Stage.COMPOSITION:
            assert state_subenv == state_composition
        elif stage == Stage.SPACE_GROUP:
            assert state_subenv == state_space_group
        elif stage == Stage.LATTICE_PARAMETERS:
            assert state_subenv == state_lattice_parameters
        else:
            raise ValueError(f"Unrecognized stage {stage}.")


@pytest.mark.parametrize(
    "states",
    [
        [
            [0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1],
            [0, 0, 4, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1],
            [1, 3, 1, 0, 6, 0, 0, 0, -1, -1, -1, -1, -1, -1],
            [2, 1, 0, 4, 0, 4, 3, 105, 0.12, 0.23, 0.34, 0.45, 0.56, 0.67],
            [1, 3, 1, 0, 6, 1, 0, 0, -1, -1, -1, -1, -1, -1],
            [1, 3, 1, 0, 6, 1, 1, 0, -1, -1, -1, -1, -1, -1],
            [0, 3, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1],
            [0, 3, 0, 0, 6, 0, 0, 0, -1, -1, -1, -1, -1, -1],
            [1, 3, 1, 0, 6, 1, 2, 0, -1, -1, -1, -1, -1, -1],
            [0, 3, 1, 0, 6, 0, 0, 0, -1, -1, -1, -1, -1, -1],
            [2, 1, 0, 4, 0, 4, 3, 105, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            [2, 1, 0, 4, 0, 4, 3, 105, 0.76, 0.75, 0.74, 0.73, 0.72, 0.71],
            [0, 0, 4, 3, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1],
        ],
    ],
)
def test__get_mask_of_subenv__returns_correct_submasks(env, states):
    # Get states from each stage and masks computed with the Crystal env.
    states_dict = {stage: [] for stage in Stage}
    masks_dict = {stage: [] for stage in Stage}
    stages = []
    for s in states:
        stage = env._get_stage(s)
        states_dict[stage].append(s)
        masks_dict[stage].append(env.get_mask_invalid_actions_forward(s))
        stages.append(stage)

    for stage, subenv in env.subenvs.items():
        # Get masks computed with subenv
        masks_subenv = tbool(
            [
                subenv.get_mask_invalid_actions_forward(
                    env._get_state_of_subenv(s, stage)
                )
                for s in states_dict[stage]
            ],
            device=env.device,
        )
        assert torch.all(
            torch.eq(
                env._get_mask_of_subenv(
                    tbool(masks_dict[stage], device=env.device), stage
                ),
                masks_subenv,
            )
        )


@pytest.mark.repeat(10)
def test__step_random__does_not_crash_from_source(env):
    """
    Very low bar test...
    """
    env.reset()
    env.step_random()
    pass


# @pytest.mark.skip(reason="skip while developping other tests")
@pytest.mark.parametrize(
    "states",
    [
        [
            [0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1],
            [0, 0, 4, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1],
            [0, 0, 4, 3, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1],
            [0, 3, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1],
            [0, 3, 0, 0, 6, 0, 0, 0, -1, -1, -1, -1, -1, -1],
            [0, 3, 1, 0, 6, 0, 0, 0, -1, -1, -1, -1, -1, -1],
        ],
        [
            [0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1],
            [1, 3, 1, 0, 6, 0, 0, 0, -1, -1, -1, -1, -1, -1],
            [1, 3, 1, 0, 6, 1, 0, 0, -1, -1, -1, -1, -1, -1],
            [1, 3, 1, 0, 6, 1, 1, 0, -1, -1, -1, -1, -1, -1],
        ],
        [
            [0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1],
            [0, 0, 4, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1],
            [1, 3, 1, 0, 6, 0, 0, 0, -1, -1, -1, -1, -1, -1],
            [2, 1, 0, 4, 0, 4, 3, 105, 0.12, 0.23, 0.34, 0.45, 0.56, 0.67],
            [1, 3, 1, 0, 6, 1, 0, 0, -1, -1, -1, -1, -1, -1],
            [1, 3, 1, 0, 6, 1, 1, 0, -1, -1, -1, -1, -1, -1],
            [0, 3, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1],
            [0, 3, 0, 0, 6, 0, 0, 0, -1, -1, -1, -1, -1, -1],
            [1, 3, 1, 0, 6, 1, 2, 0, -1, -1, -1, -1, -1, -1],
            [0, 3, 1, 0, 6, 0, 0, 0, -1, -1, -1, -1, -1, -1],
            [2, 1, 0, 4, 0, 4, 3, 105, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            [2, 1, 0, 4, 0, 4, 3, 105, 0.76, 0.75, 0.74, 0.73, 0.72, 0.71],
            [0, 0, 4, 3, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1],
        ],
    ],
)
def test__sample_actions_forward__returns_valid_actions(env, states):
    """
    Still low bar, but getting better...
    """
    n_states = len(states)
    # Get masks
    lens = [len(env.get_mask_invalid_actions_forward(s)) for s in states]
    masks = tbool(
        [env.get_mask_invalid_actions_forward(s) for s in states], device=env.device
    )
    # Build policy outputs
    params = env.random_distr_params
    policy_outputs = torch.tile(env.get_policy_output(params), dims=(n_states, 1))
    # Sample actions
    actions, _ = env.sample_actions_batch(
        policy_outputs, masks, states, is_backward=False
    )
    # Sample actions are valid
    for state, action in zip(states, actions):
        if env._get_stage(state) == Stage.LATTICE_PARAMETERS:
            continue
        assert action in env.get_valid_actions(state, done=False, backward=False)


@pytest.mark.repeat(100)
def test__trajectory_random__does_not_crash_from_source(env):
    """
    Raising the bar...
    """
    env.reset()
    env.trajectory_random()
    pass


@pytest.mark.skip(reason="skip until updated")
def test__continuous_env_common(env):
    return common.test__all_env_common(env)


@pytest.mark.skip(reason="skip until updated")
def test__all_env_common(env_with_stoichiometry_sg_check):
    return common.test__all_env_common(env_with_stoichiometry_sg_check)
