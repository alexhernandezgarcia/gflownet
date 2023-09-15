import common
import pytest
import torch
from torch import Tensor

from gflownet.envs.crystals.crystal import Crystal, Stage
from gflownet.envs.crystals.lattice_parameters import TRICLINIC


@pytest.fixture
def env():
    return Crystal(
        composition_kwargs={"elements": 4}, lattice_parameters_kwargs={"grid_size": 10}
    )


@pytest.fixture
def env_with_stoichiometry_sg_check():
    return Crystal(
        composition_kwargs={"elements": 4},
        lattice_parameters_kwargs={"grid_size": 10},
        do_stoichiometry_sg_check=True,
    )


@pytest.fixture
def env_with_space_group_stage_first():
    return Crystal(
        composition_kwargs={"elements": 4},
        lattice_parameters_kwargs={"grid_size": 10},
        do_sg_before_composition=True,
    )


@pytest.fixture
def env_with_space_group_stage_first_sg_check():
    return Crystal(
        composition_kwargs={"elements": 4},
        lattice_parameters_kwargs={"grid_size": 10},
        do_stoichiometry_sg_check=True,
        do_sg_before_composition=True,
    )


def test__environment__initializes_properly(env):
    pass


@pytest.mark.parametrize(
    "environment, initial_stage", [["env", 0], ["env_with_space_group_stage_first", 1]]
)
def test__environment__has_expected_initial_state(environment, initial_stage, request):
    environment = request.getfixturevalue(environment)
    expected_initial_state = [initial_stage] + [0] * (4 + 3 + 6)
    assert (
        environment.state == environment.source == expected_initial_state
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


@pytest.mark.parametrize("action", [(1, 1, -2, -2, -2, -2), (3, 4, -2, -2, -2, -2)])
def test__step__single_action_works(env, action):
    env.step(action)

    assert env.state != env.source


@pytest.mark.parametrize(
    "environment, actions, exp_result, exp_stage, last_action_valid",
    [
        [
            "env",
            [(1, 1, -2, -2, -2, -2), (3, 4, -2, -2, -2, -2)],
            [0, 1, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            Stage.COMPOSITION,
            True,
        ],
        [
            "env",
            [(2, 225, 3, -3, -3, -3)],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            Stage.COMPOSITION,
            False,
        ],
        [
            "env",
            [(1, 1, -2, -2, -2, -2), (3, 4, -2, -2, -2, -2), (-1, -1, -2, -2, -2, -2)],
            [1, 1, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            Stage.SPACE_GROUP,
            True,
        ],
        [
            "env",
            [
                (1, 1, -2, -2, -2, -2),
                (3, 4, -2, -2, -2, -2),
                (-1, -1, -2, -2, -2, -2),
                (2, 105, 0, -3, -3, -3),
            ],
            [1, 1, 0, 4, 0, 4, 3, 105, 0, 0, 0, 0, 0, 0],
            Stage.SPACE_GROUP,
            True,
        ],
        [
            "env",
            [
                (1, 1, -2, -2, -2, -2),
                (3, 4, -2, -2, -2, -2),
                (-1, -1, -2, -2, -2, -2),
                (2, 105, 0, -3, -3, -3),
                (2, 105, 0, -3, -3, -3),
            ],
            [1, 1, 0, 4, 0, 4, 3, 105, 0, 0, 0, 0, 0, 0],
            Stage.SPACE_GROUP,
            False,
        ],
        [
            "env",
            [
                (1, 1, -2, -2, -2, -2),
                (3, 4, -2, -2, -2, -2),
                (-1, -1, -2, -2, -2, -2),
                (2, 105, 0, -3, -3, -3),
                (-1, -1, -1, -3, -3, -3),
            ],
            [2, 1, 0, 4, 0, 4, 3, 105, 0, 0, 0, 5, 5, 5],
            Stage.LATTICE_PARAMETERS,
            True,
        ],
        [
            "env",
            [
                (1, 1, -2, -2, -2, -2),
                (3, 4, -2, -2, -2, -2),
                (-1, -1, -2, -2, -2, -2),
                (2, 105, 0, -3, -3, -3),
                (-1, -1, -1, -3, -3, -3),
                (1, 0, 0, 0, 0, 0),
            ],
            [2, 1, 0, 4, 0, 4, 3, 105, 0, 0, 0, 5, 5, 5],
            Stage.LATTICE_PARAMETERS,
            False,
        ],
        [
            "env",
            [
                (1, 1, -2, -2, -2, -2),
                (3, 4, -2, -2, -2, -2),
                (-1, -1, -2, -2, -2, -2),
                (2, 105, 0, -3, -3, -3),
                (-1, -1, -1, -3, -3, -3),
                (1, 1, 1, 0, 0, 0),
            ],
            [2, 1, 0, 4, 0, 4, 3, 105, 1, 1, 1, 5, 5, 5],
            Stage.LATTICE_PARAMETERS,
            True,
        ],
        [
            "env",
            [
                (1, 1, -2, -2, -2, -2),
                (3, 4, -2, -2, -2, -2),
                (-1, -1, -2, -2, -2, -2),
                (2, 105, 0, -3, -3, -3),
                (-1, -1, -1, -3, -3, -3),
                (1, 1, 1, 0, 0, 0),
                (0, 0, 0, 0, 0, 0),
            ],
            [2, 1, 0, 4, 0, 4, 3, 105, 1, 1, 1, 5, 5, 5],
            Stage.LATTICE_PARAMETERS,
            False,
        ],
        [
            "env",
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
            [2, 1, 0, 4, 0, 4, 3, 105, 2, 2, 1, 5, 5, 5],
            Stage.LATTICE_PARAMETERS,
            True,
        ],
        [
            "env_with_space_group_stage_first",
            [(2, 105, 0, -3, -3, -3)],
            [1, 0, 0, 0, 0, 4, 3, 105, 0, 0, 0, 0, 0, 0],
            Stage.SPACE_GROUP,
            True,
        ],
        [
            "env_with_space_group_stage_first",
            [(2, 105, 0, -3, -3, -3), (2, 105, 0, -3, -3, -3)],
            [1, 0, 0, 0, 0, 4, 3, 105, 0, 0, 0, 0, 0, 0],
            Stage.SPACE_GROUP,
            False,
        ],
        [
            "env_with_space_group_stage_first",
            [(1, 1, -2, -2, -2, -2)],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            Stage.SPACE_GROUP,
            False,
        ],
        [
            "env_with_space_group_stage_first",
            [(2, 105, 0, -3, -3, -3), (-1, -1, -1, -3, -3, -3)],
            [0, 0, 0, 0, 0, 4, 3, 105, 0, 0, 0, 5, 5, 5],
            Stage.COMPOSITION,
            True,
        ],
        [
            "env_with_space_group_stage_first",
            [
                (2, 105, 0, -3, -3, -3),
                (-1, -1, -1, -3, -3, -3),
                (1, 1, -2, -2, -2, -2),
                (3, 4, -2, -2, -2, -2),
                (3, 2, -2, -2, -2, -2),
            ],
            [0, 1, 0, 4, 0, 4, 3, 105, 0, 0, 0, 5, 5, 5],
            Stage.COMPOSITION,
            False,
        ],
        [
            "env_with_space_group_stage_first",
            [
                (2, 105, 0, -3, -3, -3),
                (-1, -1, -1, -3, -3, -3),
                (1, 1, -2, -2, -2, -2),
                (3, 4, -2, -2, -2, -2),
                (-1, -1, -2, -2, -2, -2),
            ],
            [2, 1, 0, 4, 0, 4, 3, 105, 0, 0, 0, 5, 5, 5],
            Stage.LATTICE_PARAMETERS,
            True,
        ],
        [
            "env_with_space_group_stage_first",
            [
                (2, 105, 0, -3, -3, -3),
                (-1, -1, -1, -3, -3, -3),
                (1, 1, -2, -2, -2, -2),
                (3, 4, -2, -2, -2, -2),
                (-1, -1, -2, -2, -2, -2),
                (1, 0, 0, 0, 0, 0),
            ],
            [2, 1, 0, 4, 0, 4, 3, 105, 0, 0, 0, 5, 5, 5],
            Stage.LATTICE_PARAMETERS,
            False,
        ],
        [
            "env_with_space_group_stage_first",
            [
                (2, 105, 0, -3, -3, -3),
                (-1, -1, -1, -3, -3, -3),
                (1, 1, -2, -2, -2, -2),
                (3, 4, -2, -2, -2, -2),
                (-1, -1, -2, -2, -2, -2),
                (1, 1, 1, 0, 0, 0),
                (1, 1, 0, 0, 0, 0),
                (0, 0, 0, 0, 0, 0),
            ],
            [2, 1, 0, 4, 0, 4, 3, 105, 2, 2, 1, 5, 5, 5],
            Stage.LATTICE_PARAMETERS,
            True,
        ],
    ],
)
def test__step__action_sequence_has_expected_result(
    environment, actions, exp_result, exp_stage, last_action_valid, request
):
    environment = request.getfixturevalue(environment)
    for action in actions:
        _, _, valid = environment.step(action)

    assert environment.state == exp_result
    assert environment._get_stage() == exp_stage
    assert valid == last_action_valid


@pytest.mark.parametrize("environment", ["env", "env_with_space_group_stage_first"])
def test__get_parents__returns_no_parents_in_initial_state(environment, request):
    environment = request.getfixturevalue(environment)
    return common.test__get_parents__returns_no_parents_in_initial_state(environment)


@pytest.mark.parametrize(
    "environment, actions",
    [
        ["env", [(1, 1, -2, -2, -2, -2), (3, 4, -2, -2, -2, -2)]],
        [
            "env",
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
        [
            "env_with_space_group_stage_first",
            [
                (2, 105, 0, -3, -3, -3),
                (-1, -1, -1, -3, -3, -3),
                (1, 1, -2, -2, -2, -2),
                (3, 4, -2, -2, -2, -2),
                (-1, -1, -2, -2, -2, -2),
                (1, 1, 1, 0, 0, 0),
                (1, 1, 0, 0, 0, 0),
                (0, 0, 0, 0, 0, 0),
            ],
        ],
    ],
)
def test__get_parents__contains_previous_action_after_a_step(
    environment, actions, request
):
    environment = request.getfixturevalue(environment)
    for action in actions:
        environment.step(action)
        parents, parent_actions = environment.get_parents()
        assert action in parent_actions


@pytest.mark.parametrize(
    "environment, actions",
    [
        [
            "env",
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
        [
            "env_with_space_group_stage_first",
            [
                (2, 105, 0, -3, -3, -3),
                (-1, -1, -1, -3, -3, -3),
                (1, 1, -2, -2, -2, -2),
                (3, 4, -2, -2, -2, -2),
                (-1, -1, -2, -2, -2, -2),
                (1, 1, 1, 0, 0, 0),
                (1, 1, 0, 0, 0, 0),
                (0, 0, 0, 0, 0, 0),
            ],
        ],
    ],
)
def test__reset(environment, actions, request):
    environment = request.getfixturevalue(environment)
    for action in actions:
        environment.step(action)

    assert environment.state != environment.source
    for subenv in [
        environment.composition,
        environment.space_group,
        environment.lattice_parameters,
    ]:
        assert subenv.state != subenv.source
    assert environment.lattice_parameters.lattice_system != TRICLINIC

    environment.reset()

    assert environment.state == environment.source
    for subenv in [
        environment.composition,
        environment.space_group,
        environment.lattice_parameters,
    ]:
        assert subenv.state == subenv.source
    assert environment.lattice_parameters.lattice_system == TRICLINIC


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


def test__composition_constraints(env_with_space_group_stage_first_sg_check):
    env = env_with_space_group_stage_first_sg_check

    # Pick space group 227 for which the most specific wyckoff position has
    # multiplicity 8
    env.step((2, 227, 0, -3, -3, -3))
    env.step((-1, -1, -1, -3, -3, -3))

    # Validate that the composition constraints are active by trying to add a number of
    # atoms incompatible with the space group
    for i in range(1, 8):
        _, _, valid = env.step((1, i, -2, -2, -2, -2))
        assert not valid
    for i in range(9, 16):
        _, _, valid = env.step((1, i, -2, -2, -2, -2))
        assert not valid

    # Validate that we can add a compatible number of atoms
    _, _, valid = env.step((1, 8, -2, -2, -2, -2))
    assert valid

    # Reset and pick space group 1 which has no constraint on composition
    env.reset()
    env.step((2, 1, 0, -3, -3, -3))
    env.step((-1, -1, -1, -3, -3, -3))

    # Validate that the composition constraints have been updated for the new
    # space group by adding a number of atoms incompatible with space group 227.
    _, _, valid = env.step((1, 1, -2, -2, -2, -2))
    assert valid


def test__set_state(
    env,
    env_with_stoichiometry_sg_check,
    env_with_space_group_stage_first,
    env_with_space_group_stage_first_sg_check,
):
    def reset_all_envs():
        env.reset()
        env_with_stoichiometry_sg_check.reset()
        env_with_space_group_stage_first.reset()
        env_with_space_group_stage_first_sg_check.reset()

    def set_state_and_validate(source_env, destination_env):
        destination_env.set_state(source_env.state, source_env.done)
        assert source_env.state == destination_env.state
        assert source_env.done == destination_env.done

    actions_composition = [
        (1, 2, -2, -2, -2, -2),
        (3, 4, -2, -2, -2, -2),
        (-1, -1, -2, -2, -2, -2),
    ]
    actions_spacegroup = [
        (2, 105, 0, -3, -3, -3),
        (-1, -1, -1, -3, -3, -3),
    ]
    actions_lattice_parameters = [
        (1, 1, 1, 0, 0, 0),
        (1, 1, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0),
    ]

    # Validate state transfer during environment during composition, space group and
    # lattice parameter phases
    actions = actions_composition + actions_spacegroup + actions_lattice_parameters
    for a in actions:
        env.step(a)
        set_state_and_validate(env, env_with_stoichiometry_sg_check)

    reset_all_envs()

    # Validate that an error is raised if trying to transfer a state from a
    # composition-first environment to a space-group-first environment during the
    # composition or the space group stage. However, validate that is does work
    # during the lattice parameter stage
    for a in actions_composition + actions_spacegroup[:-1]:
        env.step(a)
        try:
            with pytest.raises(ValueError):
                set_state_and_validate(env, env_with_space_group_stage_first)
        except:
            import pdb

            pdb.set_trace()

    env.step(actions_spacegroup[-1])
    for a in actions_lattice_parameters:
        env.step(a)
        set_state_and_validate(env, env_with_space_group_stage_first)

    reset_all_envs()

    # Validate that an error is raised if trying to transfer a state from a
    # space-group-first environment to a composition-first environment during the
    # composition or the space group stage. However, validate that is does work
    # during the lattice parameter stage
    for a in actions_spacegroup + actions_composition[:-1]:
        env_with_space_group_stage_first.step(a)
        with pytest.raises(ValueError):
            set_state_and_validate(env_with_space_group_stage_first, env)

    env_with_space_group_stage_first.step(actions_composition[-1])
    for a in actions_lattice_parameters:
        env_with_space_group_stage_first.step(a)
        set_state_and_validate(env_with_space_group_stage_first, env)

    reset_all_envs()

    # Validate that moving a state from a composition-first environment without
    # space-group checks to an environment with space-group checks will correctly
    # activate those. With numbers of atoms [2, 4], if the space-group checks are
    # active, the environment should refuse space group 230 but accept space group
    # 105.
    for a in actions_composition:
        env.step(a)
    set_state_and_validate(env, env_with_stoichiometry_sg_check)

    _, _, valid = env_with_stoichiometry_sg_check.step((2, 230, 0, -3, -3, -3))
    assert not valid
    _, _, valid = env_with_stoichiometry_sg_check.step((2, 105, 0, -3, -3, -3))
    assert valid

    # Before reseting the environment, validate that selecting the space group
    # 230 (which is incompatible with the composition so far) on the environment
    # without space group checks and then trying to transfer the state to the
    # environment *with* the space group checks will be recognized as invalid.
    _, _, valid = env.step((2, 230, 0, -3, -3, -3))
    assert valid
    with pytest.raises(ValueError):
        set_state_and_validate(env, env_with_stoichiometry_sg_check)

    reset_all_envs()

    # Validate that moving a state from a space-group-first environment without
    # composition checks to an environment with composition checks will correctly
    # activate those. With space group 105, if the composition checks are
    # active, the environment should refuse adding 1 atom but accept adding 2.
    for a in actions_spacegroup:
        env_with_space_group_stage_first.step(a)
    set_state_and_validate(
        env_with_space_group_stage_first, env_with_space_group_stage_first_sg_check
    )

    _, _, valid = env_with_space_group_stage_first_sg_check.step((1, 1, -2, -2, -2, -2))
    assert not valid
    _, _, valid = env_with_space_group_stage_first_sg_check.step((1, 2, -2, -2, -2, -2))
    assert valid

    # Before reseting the environment, validate that adding a single atom
    # (which is incompatible with the space group so far) on the environment
    # without composition checks and then trying to transfer the state to the
    # environment *with* the composition checks will be recognized as invalid.
    _, _, valid = env_with_space_group_stage_first.step((1, 1, -2, -2, -2, -2))
    assert valid
    with pytest.raises(ValueError):
        set_state_and_validate(
            env_with_space_group_stage_first, env_with_space_group_stage_first_sg_check
        )

    reset_all_envs()


@pytest.mark.parametrize(
    "environment",
    [
        "env",
        "env_with_stoichiometry_sg_check",
        "env_with_space_group_stage_first",
        "env_with_space_group_stage_first_sg_check",
    ],
)
def test__all_env_common(environment, request):
    environment = request.getfixturevalue(environment)
    return common.test__all_env_common(environment)
