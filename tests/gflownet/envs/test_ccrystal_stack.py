import warnings

import common
import numpy as np
import pytest
import torch
from torch import Tensor

from gflownet.envs.crystals.ccrystal_stack import CCrystal
from gflownet.envs.crystals.clattice_parameters import TRICLINIC
from gflownet.utils.common import tbool, tfloat

SG_SUBSET_ALL_CLS_PS = [
    1,
    2,
    3,
    6,
    16,
    17,
    67,
    81,
    89,
    127,
    143,
    144,
    146,
    148,
    168,
    169,
    189,
    195,
    200,
    230,
]


@pytest.fixture
def env_mini_comp_first():
    return CCrystal(
        composition_kwargs={"elements": 4},
        do_composition_to_sg_constraints=False,
        space_group_kwargs={"space_groups_subset": list(range(1, 15 + 1)) + [105]},
    )


@pytest.fixture
def env_with_stoichiometry_sg_check():
    return CCrystal(
        composition_kwargs={"elements": 4},
        do_composition_to_sg_constraints=True,
        space_group_kwargs={"space_groups_subset": SG_SUBSET_ALL_CLS_PS},
    )


@pytest.fixture
def env_sg_first():
    return CCrystal(
        composition_kwargs={"elements": 4},
        do_sg_to_composition_constraints=True,
        do_sg_before_composition=True,
    )


@pytest.mark.parametrize(
    "env", ["env_mini_comp_first", "env_with_stoichiometry_sg_check", "env_sg_first"]
)
def test__environment__initializes_properly(env, request):
    env = request.getfixturevalue(env)
    assert True


@pytest.mark.parametrize(
    "env, source",
    [
        (
            "env_mini_comp_first",
            [
                # fmt: off
                0,
                [0, 0, 0, 0],
                [0, 0, 0],
                [-1, -1, -1, -1, -1, -1],
                # fmt: on
            ],
        ),
        (
            "env_with_stoichiometry_sg_check",
            [
                # fmt: off
                0,
                [0, 0, 0, 0],
                [0, 0, 0],
                [-1, -1, -1, -1, -1, -1],
                # fmt: on
            ],
        ),
        (
            "env_sg_first",
            [
                # fmt: off
                0,
                [0, 0, 0],
                [0, 0, 0, 0],
                [-1, -1, -1, -1, -1, -1],
                # fmt: on
            ],
        ),
    ],
)
def test__source_is_expected(env, source, request):
    env = request.getfixturevalue(env)
    assert env.equal(env.source, source)


@pytest.mark.parametrize(
    "states",
    [
        [
            [0, [0, 0, 0, 0], [0, 0, 0], [-1, -1, -1, -1, -1, -1]],
            [0, [0, 4, 0, 0], [0, 0, 0], [-1, -1, -1, -1, -1, -1]],
            [1, [3, 1, 0, 6], [0, 0, 0], [-1, -1, -1, -1, -1, -1]],
            [2, [1, 0, 4, 0], [4, 3, 105], [0.12, 0.23, 0.34, 0.45, 0.56, 0.67]],
            [1, [3, 1, 0, 6], [1, 0, 0], [-1, -1, -1, -1, -1, -1]],
            [1, [3, 1, 0, 6], [1, 1, 0], [-1, -1, -1, -1, -1, -1]],
            [0, [3, 0, 0, 0], [0, 0, 0], [-1, -1, -1, -1, -1, -1]],
            [0, [3, 0, 0, 6], [0, 0, 0], [-1, -1, -1, -1, -1, -1]],
            [1, [3, 1, 0, 6], [1, 2, 0], [-1, -1, -1, -1, -1, -1]],
            [0, [3, 1, 0, 6], [0, 0, 0], [-1, -1, -1, -1, -1, -1]],
            [2, [1, 0, 4, 0], [4, 3, 105], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]],
            [2, [1, 0, 4, 0], [4, 3, 105], [0.76, 0.75, 0.74, 0.73, 0.72, 0.71]],
            [0, [0, 4, 3, 0], [0, 0, 0], [-1, -1, -1, -1, -1, -1]],
        ],
    ],
)
def test__states2policy__is_concatenation_of_subenv_states(env_mini_comp_first, states):
    env = env_mini_comp_first
    # Get policy states from the batch of states converted into each subenv
    states_dict = {stage: [] for stage in env.subenvs}
    for state in states:
        for stage in env.subenvs:
            states_dict[stage].append(env._get_state_of_subenv(state, stage))
    states_policy_dict = {
        stage: subenv.states2policy(states_dict[stage])
        for stage, subenv in env.subenvs.items()
    }
    states_policy_expected = torch.cat(
        [el for el in states_policy_dict.values()], dim=1
    )
    # Get policy states from env.states2policy
    states_policy = env.states2policy(states)
    assert torch.all(torch.eq(states_policy, states_policy_expected))


@pytest.mark.parametrize(
    "states",
    [
        [
            [0, [0, 0, 0, 0], [0, 0, 0], [-1, -1, -1, -1, -1, -1]],
            [0, [0, 4, 0, 0], [0, 0, 0], [-1, -1, -1, -1, -1, -1]],
            [1, [3, 1, 0, 6], [0, 0, 0], [-1, -1, -1, -1, -1, -1]],
            [2, [1, 0, 4, 0], [4, 3, 105], [0.12, 0.23, 0.34, 0.45, 0.56, 0.67]],
            [1, [3, 1, 0, 6], [1, 0, 0], [-1, -1, -1, -1, -1, -1]],
            [1, [3, 1, 0, 6], [1, 1, 0], [-1, -1, -1, -1, -1, -1]],
            [0, [3, 0, 0, 0], [0, 0, 0], [-1, -1, -1, -1, -1, -1]],
            [0, [3, 0, 0, 6], [0, 0, 0], [-1, -1, -1, -1, -1, -1]],
            [1, [3, 1, 0, 6], [1, 2, 0], [-1, -1, -1, -1, -1, -1]],
            [0, [3, 1, 0, 6], [0, 0, 0], [-1, -1, -1, -1, -1, -1]],
            [2, [1, 0, 4, 0], [4, 3, 105], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]],
            [2, [1, 0, 4, 0], [4, 3, 105], [0.76, 0.75, 0.74, 0.73, 0.72, 0.71]],
            [0, [0, 4, 3, 0], [0, 0, 0], [-1, -1, -1, -1, -1, -1]],
        ],
    ],
)
def test__states2proxy__is_concatenation_of_subenv_states(env_mini_comp_first, states):
    env = env_mini_comp_first
    # Get proxy states from the batch of states converted into each subenv
    states_dict = {stage: [] for stage in env.subenvs}
    for state in states:
        for stage in env.subenvs:
            states_dict[stage].append(env._get_state_of_subenv(state, stage))
    states_proxy_dict = {
        stage: subenv.states2proxy(states_dict[stage])
        for stage, subenv in env.subenvs.items()
    }
    states_proxy_expected = torch.cat([el for el in states_proxy_dict.values()], dim=1)
    # Get proxy states from env.states2proxy
    states_proxy = env.states2proxy(states)
    assert torch.all(torch.eq(states_proxy, states_proxy_expected))


@pytest.mark.parametrize(
    "state, state_composition, state_space_group, state_lattice_parameters",
    [
        [
            [0, [1, 0, 4, 0], [0, 0, 0], [-1, -1, -1, -1, -1, -1]],
            [1, 0, 4, 0],
            [0, 0, 0],
            [-1, -1, -1, -1, -1, -1],
        ],
        [
            [0, [0, 0, 0, 0], [0, 0, 0], [-1, -1, -1, -1, -1, -1]],
            [0, 0, 0, 0],
            [0, 0, 0],
            [-1, -1, -1, -1, -1, -1],
        ],
        [
            [1, [1, 0, 4, 0], [0, 0, 0], [-1, -1, -1, -1, -1, -1]],
            [1, 0, 4, 0],
            [0, 0, 0],
            [-1, -1, -1, -1, -1, -1],
        ],
        [
            [1, [1, 0, 4, 0], [4, 3, 105], [-1, -1, -1, -1, -1, -1]],
            [1, 0, 4, 0],
            [4, 3, 105],
            [-1, -1, -1, -1, -1, -1],
        ],
        [
            [1, [1, 0, 4, 0], [4, 3, 105], [-1, -1, -1, -1, -1, -1]],
            [1, 0, 4, 0],
            [4, 3, 105],
            [-1, -1, -1, -1, -1, -1],
        ],
        [
            [2, [1, 0, 4, 0], [4, 3, 105], [-1, -1, -1, -1, -1, -1]],
            [1, 0, 4, 0],
            [4, 3, 105],
            [-1, -1, -1, -1, -1, -1],
        ],
        [
            [2, [1, 0, 4, 0], [4, 3, 105], [-1, -1, -1, -1, -1, -1]],
            [1, 0, 4, 0],
            [4, 3, 105],
            [-1, -1, -1, -1, -1, -1],
        ],
        [
            [2, [1, 0, 4, 0], [4, 3, 105], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]],
            [1, 0, 4, 0],
            [4, 3, 105],
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        ],
        [
            [2, [1, 0, 4, 0], [4, 3, 105], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]],
            [1, 0, 4, 0],
            [4, 3, 105],
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        ],
        [
            [2, [1, 0, 4, 0], [4, 3, 105], [0.76, 0.75, 0.74, 0.73, 0.72, 0.71]],
            [1, 0, 4, 0],
            [4, 3, 105],
            [0.76, 0.75, 0.74, 0.73, 0.72, 0.71],
        ],
        [
            [2, [1, 0, 4, 0], [4, 3, 105], [0.76, 0.75, 0.74, 0.73, 0.72, 0.71]],
            [1, 0, 4, 0],
            [4, 3, 105],
            [0.76, 0.75, 0.74, 0.73, 0.72, 0.71],
        ],
    ],
)
def test__state_of_subenv__returns_expected(
    env_mini_comp_first,
    state,
    state_composition,
    state_space_group,
    state_lattice_parameters,
):
    env = env_mini_comp_first
    for stage in env.subenvs:
        state_subenv = env._get_state_of_subenv(state, stage)
        if stage == 0:
            assert state_subenv == state_composition
        elif stage == 1:
            assert state_subenv == state_space_group
        elif stage == 2:
            assert state_subenv == state_lattice_parameters
        else:
            raise ValueError(f"Unrecognized stage {stage}.")


@pytest.mark.parametrize(
    "env, state, dones, has_lattice_parameters, has_composition_constraints, "
    "has_spacegroup_constraints",
    [
        (
            "env_mini_comp_first",
            [0, [0, 4, 0, 0], [0, 0, 0], [-1, -1, -1, -1, -1, -1]],
            [False, False, False],
            False,
            False,
            False,
        ),
        (
            "env_mini_comp_first",
            [0, [0, 4, 3, 0], [0, 0, 0], [-1, -1, -1, -1, -1, -1]],
            [False, False, False],
            False,
            False,
            False,
        ),
        (
            "env_mini_comp_first",
            [1, [3, 1, 0, 6], [1, 2, 0], [-1, -1, -1, -1, -1, -1]],
            [True, False, False],
            True,
            False,
            False,
        ),
        (
            "env_mini_comp_first",
            [2, [1, 0, 4, 0], [4, 3, 105], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]],
            [True, True, False],
            True,
            False,
            False,
        ),
        (
            "env_with_stoichiometry_sg_check",
            [2, [4, 0, 4, 0], [4, 3, 105], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]],
            [True, True, False],
            True,
            True,
            False,
        ),
        (
            "env_sg_first",
            [0, [0, 0, 0], [0, 0, 0, 0], [-1, -1, -1, -1, -1, -1]],
            [False, False, False],
            False,
            False,
            False,
        ),
        (
            "env_sg_first",
            [0, [4, 3, 105], [0, 0, 0, 0], [-1, -1, -1, -1, -1, -1]],
            [False, False, False],
            False,
            False,
            False,
        ),
        (
            "env_sg_first",
            [1, [4, 3, 105], [3, 1, 0, 6], [-1, -1, -1, -1, -1, -1]],
            [True, False, False],
            True,
            False,
            True,
        ),
        (
            "env_sg_first",
            [2, [4, 3, 105], [1, 0, 4, 0], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]],
            [True, True, False],
            True,
            False,
            True,
        ),
    ],
)
def test__set_state__sets_state_subenvs_dones_and_constraints(
    env,
    state,
    dones,
    has_lattice_parameters,
    has_composition_constraints,
    has_spacegroup_constraints,
    request,
):
    env = request.getfixturevalue(env)
    env.set_state(state)
    # Check global state
    assert env.state == state

    # Check states of subenvs
    for stage, subenv in env.subenvs.items():
        assert subenv.state == env._get_state_of_subenv(state, stage)

    # Check dones
    for subenv, done in zip(env.subenvs.values(), dones):
        assert subenv.done == done, state

    # Check lattice parameters
    if has_lattice_parameters:
        assert env.subenvs[env.stage_spacegroup].lattice_system != "None"
        assert (
            env.subenvs[env.stage_spacegroup].lattice_system
            == env.subenvs[env.stage_latticeparameters].lattice_system
        )

    # Check composition constraints
    if has_composition_constraints:
        n_atoms = [n for n in env.subenvs[env.stage_composition].state if n > 0]
        n_atoms_compatibility_dict = env.subenvs[
            env.stage_spacegroup
        ].build_n_atoms_compatibility_dict(
            n_atoms,
            env.subenvs[env.stage_spacegroup].space_groups.keys(),
        )
        assert (
            n_atoms_compatibility_dict
            == env.subenvs[env.stage_spacegroup].n_atoms_compatibility_dict
        )

    # Check spacegroup constraints
    if has_spacegroup_constraints:
        assert (
            env.subenvs[env.stage_composition].space_group
            == env.subenvs[env.stage_spacegroup].space_group
        )


# @pytest.mark.skip(reason="skip while developping other tests")
@pytest.mark.parametrize(
    "env, state",
    [
        ("env_mini_comp_first", [0, [0, 4, 0, 0], [0, 0, 0], [-1, -1, -1, -1, -1, -1]]),
        ("env_mini_comp_first", [0, [3, 0, 0, 0], [0, 0, 0], [-1, -1, -1, -1, -1, -1]]),
        ("env_mini_comp_first", [0, [3, 0, 0, 6], [0, 0, 0], [-1, -1, -1, -1, -1, -1]]),
        ("env_mini_comp_first", [0, [3, 1, 0, 6], [0, 0, 0], [-1, -1, -1, -1, -1, -1]]),
        ("env_mini_comp_first", [1, [3, 1, 0, 6], [1, 0, 0], [-1, -1, -1, -1, -1, -1]]),
        ("env_mini_comp_first", [1, [3, 1, 0, 6], [1, 1, 0], [-1, -1, -1, -1, -1, -1]]),
        ("env_mini_comp_first", [1, [3, 1, 0, 6], [1, 2, 0], [-1, -1, -1, -1, -1, -1]]),
        ("env_mini_comp_first", [1, [3, 1, 0, 6], [1, 2, 2], [-1, -1, -1, -1, -1, -1]]),
        (
            "env_mini_comp_first",
            [2, [1, 0, 4, 0], [4, 3, 105], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]],
        ),
        (
            "env_mini_comp_first",
            [2, [1, 0, 4, 0], [4, 3, 105], [0.12, 0.23, 0.34, 0.45, 0.56, 0.67]],
        ),
        (
            "env_mini_comp_first",
            [2, [1, 0, 4, 0], [4, 3, 105], [0.76, 0.75, 0.74, 0.73, 0.72, 0.71]],
        ),
        ("env_sg_first", [0, [1, 0, 0], [0, 0, 0, 0], [-1, -1, -1, -1, -1, -1]]),
        ("env_sg_first", [0, [1, 1, 0], [0, 0, 0, 0], [-1, -1, -1, -1, -1, -1]]),
        ("env_sg_first", [0, [1, 2, 0], [0, 0, 0, 0], [-1, -1, -1, -1, -1, -1]]),
        ("env_sg_first", [0, [1, 2, 2], [0, 0, 0, 0], [-1, -1, -1, -1, -1, -1]]),
        ("env_sg_first", [1, [1, 2, 2], [3, 0, 0, 0], [-1, -1, -1, -1, -1, -1]]),
        ("env_sg_first", [1, [1, 2, 2], [3, 0, 0, 6], [-1, -1, -1, -1, -1, -1]]),
        ("env_sg_first", [1, [1, 2, 2], [3, 1, 0, 6], [-1, -1, -1, -1, -1, -1]]),
        (
            "env_sg_first",
            [2, [4, 3, 105], [1, 0, 4, 0], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]],
        ),
        (
            "env_sg_first",
            [2, [4, 3, 105], [1, 0, 4, 0], [0.12, 0.23, 0.34, 0.45, 0.56, 0.67]],
        ),
        (
            "env_sg_first",
            [2, [4, 3, 105], [1, 0, 4, 0], [0.76, 0.75, 0.74, 0.73, 0.72, 0.71]],
        ),
    ],
)
def test__get_mask_invalid_actions_backward__returns_expected_general_case(
    env, state, request
):
    env = request.getfixturevalue(env)
    stage = env._get_stage(state)
    subenv = env.subenvs[stage]
    mask = env.get_mask_invalid_actions_backward(state, done=False)
    mask_subenv = mask[3 : 3 + subenv.mask_dim]
    mask_subenv_expected = subenv.get_mask_invalid_actions_backward(
        env._get_state_of_subenv(state, stage), done=False
    )
    assert mask_subenv == mask_subenv_expected, state


@pytest.mark.parametrize(
    "env, state",
    [
        ("env_mini_comp_first", [0, [0, 0, 0, 0], [0, 0, 0], [-1, -1, -1, -1, -1, -1]]),
        ("env_mini_comp_first", [1, [3, 0, 0, 6], [0, 0, 0], [-1, -1, -1, -1, -1, -1]]),
        ("env_mini_comp_first", [1, [3, 1, 0, 6], [0, 0, 0], [-1, -1, -1, -1, -1, -1]]),
        ("env_mini_comp_first", [2, [3, 1, 0, 6], [1, 2, 2], [-1, -1, -1, -1, -1, -1]]),
        ("env_mini_comp_first", [2, [3, 1, 0, 6], [2, 1, 3], [-1, -1, -1, -1, -1, -1]]),
        ("env_sg_first", [0, [0, 0, 0], [0, 0, 0, 0], [-1, -1, -1, -1, -1, -1]]),
        ("env_sg_first", [1, [1, 2, 2], [0, 0, 0, 0], [-1, -1, -1, -1, -1, -1]]),
        ("env_sg_first", [1, [2, 1, 3], [0, 0, 0, 0], [-1, -1, -1, -1, -1, -1]]),
        ("env_sg_first", [2, [1, 2, 2], [3, 1, 0, 6], [-1, -1, -1, -1, -1, -1]]),
        ("env_sg_first", [2, [2, 1, 3], [3, 1, 0, 6], [-1, -1, -1, -1, -1, -1]]),
    ],
)
def test__get_mask_invald_actions_backward__returns_expected_stage_transition(
    env, state, request
):
    env = request.getfixturevalue(env)
    stage = env._get_stage(state)
    if stage == 0:
        assert env.equal(state, env.source)
        return
    stage -= 1
    subenv = env.subenvs[stage]
    mask = env.get_mask_invalid_actions_backward(state, done=False)
    assert mask[stage]
    mask_subenv = mask[3 : 3 + subenv.mask_dim]
    mask_subenv_expected = subenv.get_mask_invalid_actions_backward(
        env._get_state_of_subenv(state, stage), done=True
    )
    assert mask_subenv == mask_subenv_expected, state


@pytest.mark.parametrize(
    "env, action",
    [
        ("env_mini_comp_first", (1, 1, 0, 0, 0, 0, 0)),
        ("env_mini_comp_first", (3, 4, 0, 0, 0, 0, 0)),
    ],
)
def test__step__action_from_source_changes_state(env, action):
    env.step(action)

    assert env.state != env.source


@pytest.mark.repeat(10)
@pytest.mark.parametrize(
    "env", ["env_mini_comp_first", "env_with_stoichiometry_sg_check", "env_sg_first"]
)
def test__step_random__does_not_crash_from_source(env, request):
    """
    Very low bar test...
    """
    env = request.getfixturevalue(env)
    env.reset()
    env.step_random()
    pass


@pytest.mark.repeat(10)
@pytest.mark.parametrize(
    "env", ["env_mini_comp_first", "env_with_stoichiometry_sg_check", "env_sg_first"]
)
def test__trajectory_random__does_not_crash_from_source(env, request):
    """
    Raising the bar...
    """
    env = request.getfixturevalue(env)
    env.reset()
    env.trajectory_random()
    pass


@pytest.mark.skip(reason="skip while developping other tests")
def test__common__env_mini_comp_first(env_mini_comp_first):
    print(
        "\n\nCommon tests for crystal without composition <-> space group constraints\n"
    )
    return common.test__continuous_env_common(env_mini_comp_first)


@pytest.mark.skip(reason="skip while developping other tests")
def test__common__env_sg_first(env_sg_first):
    print("\n\nCommon tests for crystal with space group first\n")
    return common.test__continuous_env_common(env_sg_first)
