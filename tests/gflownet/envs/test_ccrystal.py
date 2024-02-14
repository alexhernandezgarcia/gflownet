import warnings

import common
import numpy as np
import pytest
import torch
from torch import Tensor

from gflownet.envs.crystals.ccrystal import CCrystal, Stage
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
def env():
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


def test__stage_next__returns_expected(env, env_sg_first):
    assert env._get_next_stage(None) == Stage.COMPOSITION
    assert env._get_next_stage(Stage.COMPOSITION) == Stage.SPACE_GROUP
    assert env._get_next_stage(Stage.SPACE_GROUP) == Stage.LATTICE_PARAMETERS
    assert env._get_next_stage(Stage.LATTICE_PARAMETERS) == Stage.DONE
    assert env._get_next_stage(Stage.DONE) == None

    assert env_sg_first._get_next_stage(None) == Stage.SPACE_GROUP
    assert env_sg_first._get_next_stage(Stage.SPACE_GROUP) == Stage.COMPOSITION
    assert env_sg_first._get_next_stage(Stage.COMPOSITION) == Stage.LATTICE_PARAMETERS
    assert env_sg_first._get_next_stage(Stage.LATTICE_PARAMETERS) == Stage.DONE
    assert env_sg_first._get_next_stage(Stage.DONE) == None


def test__stage_prev__returns_expected(env, env_sg_first):
    assert env._get_previous_stage(Stage.COMPOSITION) == Stage.DONE
    assert env._get_previous_stage(Stage.SPACE_GROUP) == Stage.COMPOSITION
    assert env._get_previous_stage(Stage.LATTICE_PARAMETERS) == Stage.SPACE_GROUP
    assert env._get_previous_stage(Stage.DONE) == Stage.LATTICE_PARAMETERS

    assert env_sg_first._get_previous_stage(Stage.SPACE_GROUP) == Stage.DONE
    assert env_sg_first._get_previous_stage(Stage.COMPOSITION) == Stage.SPACE_GROUP
    assert (
        env_sg_first._get_previous_stage(Stage.LATTICE_PARAMETERS) == Stage.COMPOSITION
    )
    assert env_sg_first._get_previous_stage(Stage.DONE) == Stage.LATTICE_PARAMETERS


def test__environment__initializes_properly(env):
    pass


@pytest.mark.parametrize("env_input, initial_stage", [["env", 0], ["env_sg_first", 1]])
def test__environment__has_expected_initial_state(env_input, initial_stage, request):
    """
    The source of the composition and space group environments is all 0s. The source of
    the continuous lattice parameters environment is all -1s.
    """
    env = request.getfixturevalue(env_input)
    expected_initial_state = [initial_stage] + [0] * (4 + 3) + [-1] * 6
    assert (
        env.state == env.source == expected_initial_state
    )  # stage + n elements + space groups + lattice parameters


def test__environment__has_expected_action_space(env):
    assert len(env.action_space) == len(
        env.subenvs[Stage.COMPOSITION].action_space
    ) + len(env.subenvs[Stage.SPACE_GROUP].action_space) + len(
        env.subenvs[Stage.LATTICE_PARAMETERS].action_space
    )

    underlying_action_space = (
        env.subenvs[Stage.COMPOSITION].action_space
        + env.subenvs[Stage.SPACE_GROUP].action_space
        + env.subenvs[Stage.LATTICE_PARAMETERS].action_space
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
def test__states2policy__is_concatenation_of_subenv_states(env, states):
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
    states_torch = tfloat(states, float_type=env.float, device=env.device)
    states_policy = env.states2policy(states_torch)
    assert torch.all(torch.eq(states_policy, states_policy_expected))


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
def test__states2proxy__is_concatenation_of_subenv_states(env, states):
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
    states_torch = tfloat(states, float_type=env.float, device=env.device)
    states_proxy = env.states2proxy(states_torch)
    assert torch.all(torch.eq(states_proxy, states_proxy_expected))


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
def test__state2readable__is_concatenation_of_subenv_states(env, states):
    # Get policy states from the batch of states converted into each subenv
    states_readable_expected = []
    for state in states:
        readables = []
        for stage, subenv in env.subenvs.items():
            readables.append(
                subenv.state2readable(env._get_state_of_subenv(state, stage))
            )
        states_readable_expected.append(
            f"{env._get_stage(state)}; "
            f"Composition = {readables[0]}; "
            f"SpaceGroup = {readables[1]}; "
            f"LatticeParameters = {readables[2]}"
        )
    # Get policy states from env.states2policy
    states_readable = [env.state2readable(state) for state in states]
    for readable, readable_expected in zip(states_readable, states_readable_expected):
        assert readable == readable_expected


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
    "env_input, state, dones, has_lattice_parameters, has_composition_constraints, has_spacegroup_constraints",
    [
        (
            "env",
            [0, 0, 4, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1],
            [False, False, False],
            False,
            False,
            False,
        ),
        (
            "env",
            [0, 0, 4, 3, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1],
            [False, False, False],
            False,
            False,
            False,
        ),
        (
            "env",
            [1, 3, 1, 0, 6, 1, 2, 0, -1, -1, -1, -1, -1, -1],
            [True, False, False],
            True,
            False,
            False,
        ),
        (
            "env",
            [2, 1, 0, 4, 0, 4, 3, 105, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            [True, True, False],
            True,
            False,
            False,
        ),
        (
            "env_with_stoichiometry_sg_check",
            [2, 4, 0, 4, 0, 4, 3, 105, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            [True, True, False],
            True,
            True,
            False,
        ),
        (
            "env_sg_first",
            [1, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1],
            [False, False, False],
            False,
            False,
            False,
        ),
        (
            "env_sg_first",
            [1, 0, 0, 0, 0, 4, 3, 105, -1, -1, -1, -1, -1, -1],
            [False, False, False],
            False,
            False,
            False,
        ),
        (
            "env_sg_first",
            [0, 3, 1, 0, 6, 4, 3, 105, -1, -1, -1, -1, -1, -1],
            [False, True, False],
            True,
            False,
            True,
        ),
        (
            "env_sg_first",
            [2, 1, 0, 4, 0, 4, 3, 105, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            [True, True, False],
            True,
            False,
            True,
        ),
    ],
)
def test__set_state__sets_state_subenvs_dones_and_constraints(
    env_input,
    state,
    dones,
    has_lattice_parameters,
    has_composition_constraints,
    has_spacegroup_constraints,
    request,
):
    env = request.getfixturevalue(env_input)
    env.set_state(state)
    # Check global state
    assert env.state == state

    # Check states of subenvs
    for stage, subenv in env.subenvs.items():
        assert subenv.state == env._get_state_of_subenv(state, stage)

    # Check dones
    for subenv, done in zip(env.subenvs.values(), dones):
        assert subenv.done == done

    # Check lattice parameters
    if has_lattice_parameters:
        assert env.subenvs[Stage.SPACE_GROUP].lattice_system != "None"
        assert (
            env.subenvs[Stage.SPACE_GROUP].lattice_system
            == env.subenvs[Stage.LATTICE_PARAMETERS].lattice_system
        )

    # Check composition constraints
    if has_composition_constraints:
        n_atoms = [n for n in env.subenvs[Stage.COMPOSITION].state if n > 0]
        n_atoms_compatibility_dict = env.subenvs[
            Stage.SPACE_GROUP
        ].build_n_atoms_compatibility_dict(
            n_atoms,
            env.subenvs[Stage.SPACE_GROUP].space_groups.keys(),
        )
        assert (
            n_atoms_compatibility_dict
            == env.subenvs[Stage.SPACE_GROUP].n_atoms_compatibility_dict
        )

    # Check spacegroup constraints
    if has_spacegroup_constraints:
        assert (
            env.subenvs[Stage.COMPOSITION].space_group
            == env.subenvs[Stage.SPACE_GROUP].space_group
        )


@pytest.mark.parametrize(
    "env_input, state",
    [
        ("env", [0, 0, 4, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1]),
        ("env", [0, 3, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1]),
        ("env", [0, 3, 0, 0, 6, 0, 0, 0, -1, -1, -1, -1, -1, -1]),
        ("env", [0, 3, 1, 0, 6, 0, 0, 0, -1, -1, -1, -1, -1, -1]),
        ("env", [1, 3, 1, 0, 6, 1, 0, 0, -1, -1, -1, -1, -1, -1]),
        ("env", [1, 3, 1, 0, 6, 1, 1, 0, -1, -1, -1, -1, -1, -1]),
        ("env", [1, 3, 1, 0, 6, 1, 2, 0, -1, -1, -1, -1, -1, -1]),
        ("env", [1, 3, 1, 0, 6, 1, 2, 2, -1, -1, -1, -1, -1, -1]),
        ("env", [2, 1, 0, 4, 0, 4, 3, 105, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        ("env", [2, 1, 0, 4, 0, 4, 3, 105, 0.12, 0.23, 0.34, 0.45, 0.56, 0.67]),
        ("env", [2, 1, 0, 4, 0, 4, 3, 105, 0.76, 0.75, 0.74, 0.73, 0.72, 0.71]),
        ("env_sg_first", [1, 0, 0, 0, 0, 1, 0, 0, -1, -1, -1, -1, -1, -1]),
        ("env_sg_first", [1, 0, 0, 0, 0, 1, 1, 0, -1, -1, -1, -1, -1, -1]),
        ("env_sg_first", [1, 0, 0, 0, 0, 1, 2, 0, -1, -1, -1, -1, -1, -1]),
        ("env_sg_first", [1, 0, 0, 0, 0, 1, 2, 2, -1, -1, -1, -1, -1, -1]),
        ("env_sg_first", [0, 3, 0, 0, 0, 1, 2, 2, -1, -1, -1, -1, -1, -1]),
        ("env_sg_first", [0, 3, 0, 0, 6, 1, 2, 2, -1, -1, -1, -1, -1, -1]),
        ("env_sg_first", [0, 3, 1, 0, 6, 1, 2, 2, -1, -1, -1, -1, -1, -1]),
        ("env_sg_first", [2, 1, 0, 4, 0, 4, 3, 105, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        (
            "env_sg_first",
            [2, 1, 0, 4, 0, 4, 3, 105, 0.12, 0.23, 0.34, 0.45, 0.56, 0.67],
        ),
        (
            "env_sg_first",
            [2, 1, 0, 4, 0, 4, 3, 105, 0.76, 0.75, 0.74, 0.73, 0.72, 0.71],
        ),
    ],
)
def test__get_mask_invalid_actions_backward__returns_expected_general_case(
    env_input, state, request
):
    env = request.getfixturevalue(env_input)
    stage = env._get_stage(state)
    mask = env.get_mask_invalid_actions_backward(state, done=False)
    for stg, subenv in env.subenvs.items():
        if stg == stage:
            # Mask of state if stage is current stage in state
            mask_subenv_expected = subenv.get_mask_invalid_actions_backward(
                env._get_state_of_subenv(state, stg)
            )
        else:
            # Mask of source if stage is other than current stage in state
            mask_subenv_expected = subenv.get_mask_invalid_actions_backward(
                subenv.source
            )
        mask_subenv = env._get_mask_of_subenv(mask, stg)
        assert mask_subenv == mask_subenv_expected


@pytest.mark.parametrize(
    "env_input, state",
    [
        ("env", [0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1]),
        ("env", [1, 3, 0, 0, 6, 0, 0, 0, -1, -1, -1, -1, -1, -1]),
        ("env", [1, 3, 1, 0, 6, 0, 0, 0, -1, -1, -1, -1, -1, -1]),
        ("env", [2, 3, 1, 0, 6, 1, 2, 2, -1, -1, -1, -1, -1, -1]),
        ("env", [2, 3, 1, 0, 6, 2, 1, 3, -1, -1, -1, -1, -1, -1]),
        ("env_sg_first", [1, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1]),
        ("env_sg_first", [0, 0, 0, 0, 0, 1, 2, 2, -1, -1, -1, -1, -1, -1]),
        ("env_sg_first", [0, 0, 0, 0, 0, 2, 1, 3, -1, -1, -1, -1, -1, -1]),
        ("env_sg_first", [2, 3, 1, 0, 6, 1, 2, 2, -1, -1, -1, -1, -1, -1]),
        ("env_sg_first", [2, 3, 1, 0, 6, 2, 1, 3, -1, -1, -1, -1, -1, -1]),
    ],
)
def test__get_mask_invald_actions_backward__returns_expected_stage_transition(
    env_input, state, request
):
    env = request.getfixturevalue(env_input)
    stage = env._get_stage(state)
    prev_stage = env._get_previous_stage(stage)
    mask = env.get_mask_invalid_actions_backward(state, done=False)
    for stg, subenv in env.subenvs.items():
        if stg == prev_stage and prev_stage != Stage.DONE:
            # Mask of done (EOS only) if stage is previous stage in state
            mask_subenv_expected = subenv.get_mask_invalid_actions_backward(
                env._get_state_of_subenv(state, stg), done=True
            )
        else:
            mask_subenv_expected = subenv.get_mask_invalid_actions_backward(
                subenv.source
            )
            if stg == stage:
                assert env._get_state_of_subenv(state, stg) == subenv.source
        mask_subenv = env._get_mask_of_subenv(mask, stg)
        assert mask_subenv == mask_subenv_expected


@pytest.mark.parametrize(
    "action", [(1, 1, -2, -2, -2, -2, -2), (3, 4, -2, -2, -2, -2, -2)]
)
def test__step__single_action_works(env, action):
    env.step(action)

    assert env.state != env.source


@pytest.mark.parametrize(
    "env_input, actions, exp_result, exp_stage, last_action_valid",
    [
        [
            "env",
            [(1, 1, -2, -2, -2, -2, -2), (3, 4, -2, -2, -2, -2, -2)],
            [0, 1, 0, 4, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1],
            Stage.COMPOSITION,
            True,
        ],
        [
            "env",
            [(2, 105, 3, -3, -3, -3, -3)],
            [0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1],
            Stage.COMPOSITION,
            False,
        ],
        [
            "env",
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
            "env",
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
            "env",
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
            "env",
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
            "env",
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
            "env",
            [
                (1, 1, -2, -2, -2, -2, -2),
                (3, 4, -2, -2, -2, -2, -2),
                (-1, -1, -2, -2, -2, -2, -2),
                (2, 105, 0, -3, -3, -3, -3),
                (-1, -1, -1, -3, -3, -3, -3),
                (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1),
            ],
            [2, 1, 0, 4, 0, 4, 3, 105, 0.1, 0.1, 0.3, 0.4, 0.4, 0.4],
            Stage.LATTICE_PARAMETERS,
            True,
        ],
        [
            "env",
            [
                (1, 1, -2, -2, -2, -2, -2),
                (3, 4, -2, -2, -2, -2, -2),
                (-1, -1, -2, -2, -2, -2, -2),
                (2, 105, 0, -3, -3, -3, -3),
                (-1, -1, -1, -3, -3, -3, -3),
                (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1),
                (0.6, 0.5, 0.8, 0.3, 0.2, 0.6, 0),
            ],
            [2, 1, 0, 4, 0, 4, 3, 105, 0.1, 0.1, 0.3, 0.4, 0.4, 0.4],
            Stage.LATTICE_PARAMETERS,
            False,
        ],
        [
            "env",
            [
                (1, 1, -2, -2, -2, -2, -2),
                (3, 4, -2, -2, -2, -2, -2),
                (-1, -1, -2, -2, -2, -2, -2),
                (2, 105, 0, -3, -3, -3, -3),
                (-1, -1, -1, -3, -3, -3, -3),
                (0.1, 0.1, 0.3, 0.0, 0.0, 0.0, 1),
                (0.66, 0.0, 0.44, 0.0, 0.0, 0.0, 0),
            ],
            [2, 1, 0, 4, 0, 4, 3, 105, 0.76, 0.76, 0.74, 0.4, 0.4, 0.4],
            Stage.LATTICE_PARAMETERS,
            True,
        ],
        [
            "env",
            [
                (1, 1, -2, -2, -2, -2, -2),
                (3, 4, -2, -2, -2, -2, -2),
                (-1, -1, -2, -2, -2, -2, -2),
                (2, 105, 0, -3, -3, -3, -3),
                (-1, -1, -1, -3, -3, -3, -3),
                (0.1, 0.1, 0.3, 0.0, 0.0, 0.0, 1),
                (0.66, 0.66, 0.44, 0.0, 0.0, 0.0, 0),
                (np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf),
            ],
            [2, 1, 0, 4, 0, 4, 3, 105, 0.76, 0.76, 0.74, 0.4, 0.4, 0.4],
            Stage.LATTICE_PARAMETERS,
            True,
        ],
        [
            "env_sg_first",
            [(1, 1, -2, -2, -2, -2, -2)],
            [1, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1],
            Stage.SPACE_GROUP,
            False,
        ],
        [
            "env_sg_first",
            [(2, 105, 0, -3, -3, -3, -3)],
            [1, 0, 0, 0, 0, 4, 3, 105, -1, -1, -1, -1, -1, -1],
            Stage.SPACE_GROUP,
            True,
        ],
        [
            "env_sg_first",
            [(2, 105, 0, -3, -3, -3, -3), (2, 105, 0, -3, -3, -3, -3)],
            [1, 0, 0, 0, 0, 4, 3, 105, -1, -1, -1, -1, -1, -1],
            Stage.SPACE_GROUP,
            False,
        ],
        [
            "env_sg_first",
            [(2, 105, 0, -3, -3, -3, -3), (-1, -1, -1, -3, -3, -3, -3)],
            [0, 0, 0, 0, 0, 4, 3, 105, -1, -1, -1, -1, -1, -1],
            Stage.COMPOSITION,
            True,
        ],
        [
            "env_sg_first",
            [
                (2, 105, 0, -3, -3, -3, -3),
                (-1, -1, -1, -3, -3, -3, -3),
                (3, 4, -2, -2, -2, -2, -2),
            ],
            [0, 0, 0, 4, 0, 4, 3, 105, -1, -1, -1, -1, -1, -1],
            Stage.COMPOSITION,
            True,
        ],
        [
            "env_sg_first",
            [
                (2, 105, 0, -3, -3, -3, -3),
                (-1, -1, -1, -3, -3, -3, -3),
                (1, 2, -2, -2, -2, -2, -2),
                (3, 4, -2, -2, -2, -2, -2),
                (-1, -1, -2, -2, -2, -2, -2),
            ],
            [2, 2, 0, 4, 0, 4, 3, 105, -1, -1, -1, -1, -1, -1],
            Stage.LATTICE_PARAMETERS,
            True,
        ],
        [
            "env_sg_first",
            [
                (2, 105, 0, -3, -3, -3, -3),
                (-1, -1, -1, -3, -3, -3, -3),
                (1, 2, -2, -2, -2, -2, -2),
                (3, 4, -2, -2, -2, -2, -2),
                (-1, -1, -2, -2, -2, -2, -2),
                (1.5, 0, 0, 0, 0, 0, 0),
            ],
            [2, 2, 0, 4, 0, 4, 3, 105, -1, -1, -1, -1, -1, -1],
            Stage.LATTICE_PARAMETERS,
            False,
        ],
        [
            "env_sg_first",
            [
                (2, 105, 0, -3, -3, -3, -3),
                (-1, -1, -1, -3, -3, -3, -3),
                (1, 2, -2, -2, -2, -2, -2),
                (3, 4, -2, -2, -2, -2, -2),
                (-1, -1, -2, -2, -2, -2, -2),
                (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1),
                (0.6, 0.5, 0.8, 0.3, 0.2, 0.6, 0),
            ],
            [2, 2, 0, 4, 0, 4, 3, 105, 0.1, 0.1, 0.3, 0.4, 0.4, 0.4],
            Stage.LATTICE_PARAMETERS,
            False,
        ],
        [
            "env_sg_first",
            [
                (2, 105, 0, -3, -3, -3, -3),
                (-1, -1, -1, -3, -3, -3, -3),
                (1, 2, -2, -2, -2, -2, -2),
                (3, 4, -2, -2, -2, -2, -2),
                (-1, -1, -2, -2, -2, -2, -2),
                (0.1, 0.1, 0.3, 0.0, 0.0, 0.0, 1),
                (0.66, 0.0, 0.44, 0.0, 0.0, 0.0, 0),
            ],
            [2, 2, 0, 4, 0, 4, 3, 105, 0.76, 0.76, 0.74, 0.4, 0.4, 0.4],
            Stage.LATTICE_PARAMETERS,
            True,
        ],
        [
            "env_sg_first",
            [
                (2, 105, 0, -3, -3, -3, -3),
                (-1, -1, -1, -3, -3, -3, -3),
                (1, 2, -2, -2, -2, -2, -2),
                (3, 4, -2, -2, -2, -2, -2),
                (-1, -1, -2, -2, -2, -2, -2),
                (0.1, 0.1, 0.3, 0.0, 0.0, 0.0, 1),
                (0.66, 0.66, 0.44, 0.0, 0.0, 0.0, 0),
                (np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf),
            ],
            [2, 2, 0, 4, 0, 4, 3, 105, 0.76, 0.76, 0.74, 0.4, 0.4, 0.4],
            Stage.LATTICE_PARAMETERS,
            True,
        ],
    ],
)
def test__step__action_sequence_has_expected_result(
    env_input, actions, exp_result, exp_stage, last_action_valid, request
):
    env = request.getfixturevalue(env_input)
    for action in actions:
        warnings.filterwarnings("ignore")
        _, _, valid = env.step(action)

    assert env.state == exp_result
    assert env._get_stage() == exp_stage
    assert valid == last_action_valid


@pytest.mark.parametrize(
    "env_input, state_init, state_end, stage_init, stage_end, actions, last_action_valid",
    [
        [
            "env",
            [0, 1, 0, 4, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1],
            [0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1],
            Stage.COMPOSITION,
            Stage.COMPOSITION,
            [(3, 4, -2, -2, -2, -2, -2), (1, 1, -2, -2, -2, -2, -2)],
            True,
        ],
        [
            "env",
            [0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1],
            [0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1],
            Stage.COMPOSITION,
            Stage.COMPOSITION,
            [(2, 105, 3, -3, -3, -3, -3)],
            False,
        ],
        [
            "env",
            [1, 1, 0, 4, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1],
            [0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1],
            Stage.SPACE_GROUP,
            Stage.COMPOSITION,
            [
                (-1, -1, -2, -2, -2, -2, -2),
                (3, 4, -2, -2, -2, -2, -2),
                (1, 1, -2, -2, -2, -2, -2),
            ],
            True,
        ],
        [
            "env",
            [1, 1, 0, 4, 0, 4, 3, 105, -1, -1, -1, -1, -1, -1],
            [0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1],
            Stage.SPACE_GROUP,
            Stage.COMPOSITION,
            [
                (2, 105, 0, -3, -3, -3, -3),
                (-1, -1, -2, -2, -2, -2, -2),
                (3, 4, -2, -2, -2, -2, -2),
                (1, 1, -2, -2, -2, -2, -2),
            ],
            True,
        ],
        [
            "env",
            [2, 1, 0, 4, 0, 4, 3, 105, -1, -1, -1, -1, -1, -1],
            [0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1],
            Stage.LATTICE_PARAMETERS,
            Stage.COMPOSITION,
            [
                (-1, -1, -1, -3, -3, -3, -3),
                (2, 105, 0, -3, -3, -3, -3),
                (-1, -1, -2, -2, -2, -2, -2),
                (3, 4, -2, -2, -2, -2, -2),
                (1, 1, -2, -2, -2, -2, -2),
            ],
            True,
        ],
        [
            "env",
            [2, 1, 0, 4, 0, 4, 3, 105, 0.1, 0.1, 0.3, 0.4, 0.4, 0.4],
            [2, 1, 0, 4, 0, 4, 3, 105, 0.1, 0.1, 0.3, 0.4, 0.4, 0.4],
            Stage.LATTICE_PARAMETERS,
            Stage.LATTICE_PARAMETERS,
            [
                (1.5, 0, 0, 0, 0, 0, 0),
            ],
            False,
        ],
        [
            "env",
            [2, 1, 0, 4, 0, 4, 3, 105, 0.1, 0.1, 0.3, 0.4, 0.4, 0.4],
            [0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1],
            Stage.LATTICE_PARAMETERS,
            Stage.COMPOSITION,
            [
                (0.1, 0.1, 0.3, 0.0, 0.0, 0.0, 1),
                (-1, -1, -1, -3, -3, -3, -3),
                (2, 105, 0, -3, -3, -3, -3),
                (-1, -1, -2, -2, -2, -2, -2),
                (3, 4, -2, -2, -2, -2, -2),
                (1, 1, -2, -2, -2, -2, -2),
            ],
            True,
        ],
        [
            "env",
            [2, 1, 0, 4, 0, 4, 3, 105, 0.76, 0.76, 0.74, 0.4, 0.4, 0.4],
            [0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1],
            Stage.LATTICE_PARAMETERS,
            Stage.COMPOSITION,
            [
                (0.66, 0.0, 0.44, 0.0, 0.0, 0.0, 0),
                (0.1, 0.1, 0.3, 0.0, 0.0, 0.0, 1),
                (-1, -1, -1, -3, -3, -3, -3),
                (2, 105, 0, -3, -3, -3, -3),
                (-1, -1, -2, -2, -2, -2, -2),
                (3, 4, -2, -2, -2, -2, -2),
                (1, 1, -2, -2, -2, -2, -2),
            ],
            True,
        ],
        [
            "env",
            [2, 1, 0, 4, 0, 4, 3, 105, 0.76, 0.76, 0.74, 0.4, 0.4, 0.4],
            [0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1],
            Stage.LATTICE_PARAMETERS,
            Stage.COMPOSITION,
            [
                (np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf),
                (0.66, 0.0, 0.44, 0.0, 0.0, 0.0, 0),
                (0.1, 0.1, 0.3, 0.0, 0.0, 0.0, 1),
                (-1, -1, -1, -3, -3, -3, -3),
                (2, 105, 0, -3, -3, -3, -3),
                (-1, -1, -2, -2, -2, -2, -2),
                (3, 4, -2, -2, -2, -2, -2),
                (1, 1, -2, -2, -2, -2, -2),
            ],
            True,
        ],
        [
            "env_sg_first",
            [1, 0, 0, 0, 0, 4, 3, 105, -1, -1, -1, -1, -1, -1],
            [1, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1],
            Stage.SPACE_GROUP,
            Stage.SPACE_GROUP,
            [
                (2, 105, 0, -3, -3, -3, -3),
            ],
            True,
        ],
        [
            "env_sg_first",
            [0, 1, 0, 4, 0, 4, 3, 105, -1, -1, -1, -1, -1, -1],
            [1, 0, 0, 0, 0, 4, 3, 105, -1, -1, -1, -1, -1, -1],
            Stage.COMPOSITION,
            Stage.SPACE_GROUP,
            [
                (3, 4, -2, -2, -2, -2, -2),
                (1, 1, -2, -2, -2, -2, -2),
                (-1, -1, -1, -3, -3, -3, -3),
            ],
            True,
        ],
        [
            "env_sg_first",
            [2, 1, 0, 4, 0, 4, 3, 105, -1, -1, -1, -1, -1, -1],
            [1, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1],
            Stage.LATTICE_PARAMETERS,
            Stage.SPACE_GROUP,
            [
                (-1, -1, -2, -2, -2, -2, -2),
                (3, 4, -2, -2, -2, -2, -2),
                (1, 1, -2, -2, -2, -2, -2),
                (-1, -1, -1, -3, -3, -3, -3),
                (2, 105, 0, -3, -3, -3, -3),
            ],
            True,
        ],
        [
            "env_sg_first",
            [2, 1, 0, 4, 0, 4, 3, 105, 0.1, 0.1, 0.3, 0.4, 0.4, 0.4],
            [2, 1, 0, 4, 0, 4, 3, 105, 0.1, 0.1, 0.3, 0.4, 0.4, 0.4],
            Stage.LATTICE_PARAMETERS,
            Stage.LATTICE_PARAMETERS,
            [
                (1.5, 0, 0, 0, 0, 0, 0),
            ],
            False,
        ],
        [
            "env_sg_first",
            [2, 1, 0, 4, 0, 4, 3, 105, 0.76, 0.76, 0.74, 0.4, 0.4, 0.4],
            [1, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1],
            Stage.LATTICE_PARAMETERS,
            Stage.SPACE_GROUP,
            [
                (np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf),
                (0.66, 0.0, 0.44, 0.0, 0.0, 0.0, 0),
                (0.1, 0.1, 0.3, 0.0, 0.0, 0.0, 1),
                (-1, -1, -2, -2, -2, -2, -2),
                (3, 4, -2, -2, -2, -2, -2),
                (1, 1, -2, -2, -2, -2, -2),
                (-1, -1, -1, -3, -3, -3, -3),
                (2, 105, 0, -3, -3, -3, -3),
            ],
            True,
        ],
    ],
)
def test__step_backwards__action_sequence_has_expected_result(
    env_input,
    state_init,
    state_end,
    stage_init,
    stage_end,
    actions,
    last_action_valid,
    request,
):
    env = request.getfixturevalue(env_input)

    # Hacky way to also test if first action global EOS
    if actions[0] == env.eos:
        env.set_state(state_init, done=True)
    else:
        env.set_state(state_init, done=False)
    assert env.state == state_init
    assert env._get_stage() == stage_init
    for action in actions:
        warnings.filterwarnings("ignore")
        _, _, valid = env.step_backwards(action)

    assert env.state == state_end
    assert env._get_stage() == stage_end
    assert valid == last_action_valid


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
    for subenv in [
        env.subenvs[Stage.COMPOSITION],
        env.subenvs[Stage.SPACE_GROUP],
        env.subenvs[Stage.LATTICE_PARAMETERS],
    ]:
        assert subenv.state != subenv.source
    assert env.subenvs[Stage.LATTICE_PARAMETERS].lattice_system != TRICLINIC

    env.reset()

    assert env.state == env.source
    for subenv in [
        env.subenvs[Stage.COMPOSITION],
        env.subenvs[Stage.SPACE_GROUP],
        env.subenvs[Stage.LATTICE_PARAMETERS],
    ]:
        assert subenv.state == subenv.source
    assert env.subenvs[Stage.LATTICE_PARAMETERS].lattice_system == TRICLINIC


# TODO: write new test of masks, both fw and bw
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
        assert not all(mask[: len(env.subenvs[Stage.COMPOSITION].action_space)])
        assert all(mask[len(env.subenvs[Stage.COMPOSITION].action_space) :])
    if env._get_stage() == Stage.SPACE_GROUP:
        assert not all(
            mask[
                len(env.subenvs[Stage.COMPOSITION].action_space) : len(
                    env.subenvs[Stage.COMPOSITION].action_space
                )
                + len(env.subenvs[Stage.SPACE_GROUP].action_space)
            ]
        )
        assert all(mask[: len(env.subenvs[Stage.COMPOSITION].action_space)])
        assert all(
            mask[
                len(env.subenvs[Stage.COMPOSITION].action_space)
                + len(env.subenvs[Stage.SPACE_GROUP].action_space) :
            ]
        )
    if env._get_stage() == Stage.LATTICE_PARAMETERS:
        assert not all(
            mask[
                len(env.subenvs[Stage.COMPOSITION].action_space)
                + len(env.subenvs[Stage.SPACE_GROUP].action_space) :
            ]
        )
        assert all(
            mask[
                : len(env.subenvs[Stage.COMPOSITION].action_space)
                + len(env.subenvs[Stage.SPACE_GROUP].action_space)
            ]
        )


def test__get_policy_outputs__is_the_concatenation_of_subenvs(env):
    policy_output_composition = env.subenvs[Stage.COMPOSITION].get_policy_output(
        env.subenvs[Stage.COMPOSITION].fixed_distr_params
    )
    policy_output_space_group = env.subenvs[Stage.SPACE_GROUP].get_policy_output(
        env.subenvs[Stage.SPACE_GROUP].fixed_distr_params
    )
    policy_output_lattice_parameters = env.subenvs[
        Stage.LATTICE_PARAMETERS
    ].get_policy_output(env.subenvs[Stage.LATTICE_PARAMETERS].fixed_distr_params)
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
        env.subenvs[Stage.COMPOSITION].get_policy_output(
            env.subenvs[Stage.COMPOSITION].fixed_distr_params
        ),
        dims=(n_states, 1),
    )
    policy_output_space_group = torch.tile(
        env.subenvs[Stage.SPACE_GROUP].get_policy_output(
            env.subenvs[Stage.SPACE_GROUP].fixed_distr_params
        ),
        dims=(n_states, 1),
    )
    policy_output_lattice_parameters = torch.tile(
        env.subenvs[Stage.LATTICE_PARAMETERS].get_policy_output(
            env.subenvs[Stage.LATTICE_PARAMETERS].fixed_distr_params
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
            [2, 1, 0, 4, 0, 4, 3, 105, 0.1, 0.1, 0.3, 0.4, 0.4, 0.4],
            [1, 3, 1, 0, 6, 1, 0, 0, -1, -1, -1, -1, -1, -1],
            [1, 3, 1, 0, 6, 1, 1, 0, -1, -1, -1, -1, -1, -1],
            [0, 3, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1],
            [0, 3, 0, 0, 6, 0, 0, 0, -1, -1, -1, -1, -1, -1],
            [1, 3, 1, 0, 6, 1, 2, 0, -1, -1, -1, -1, -1, -1],
            [0, 3, 1, 0, 6, 0, 0, 0, -1, -1, -1, -1, -1, -1],
            [2, 1, 0, 4, 0, 4, 3, 105, 0.5, 0.5, 0.3, 0.4, 0.4, 0.4],
            [2, 1, 0, 4, 0, 4, 3, 105, 0.45, 0.45, 0.33, 0.4, 0.4, 0.4],
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


@pytest.mark.parametrize(
    "states",
    [
        [
            [0, 0, 4, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1],
            [0, 0, 4, 3, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1],
            [0, 3, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1],
            [0, 3, 0, 0, 6, 0, 0, 0, -1, -1, -1, -1, -1, -1],
            [0, 3, 1, 0, 6, 0, 0, 0, -1, -1, -1, -1, -1, -1],
        ],
        [
            [1, 3, 1, 0, 6, 0, 0, 0, -1, -1, -1, -1, -1, -1],
            [1, 3, 1, 0, 6, 1, 0, 0, -1, -1, -1, -1, -1, -1],
            [1, 3, 1, 0, 6, 1, 1, 0, -1, -1, -1, -1, -1, -1],
        ],
        [
            [0, 0, 4, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1],
            [1, 3, 1, 0, 6, 0, 0, 0, -1, -1, -1, -1, -1, -1],
            [2, 1, 0, 4, 0, 4, 3, 105, 0.1, 0.1, 0.3, 0.4, 0.4, 0.4],
            [1, 3, 1, 0, 6, 1, 0, 0, -1, -1, -1, -1, -1, -1],
            [1, 3, 1, 0, 6, 1, 1, 0, -1, -1, -1, -1, -1, -1],
            [0, 3, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1],
            [0, 3, 0, 0, 6, 0, 0, 0, -1, -1, -1, -1, -1, -1],
            [1, 3, 1, 0, 6, 1, 2, 0, -1, -1, -1, -1, -1, -1],
            [0, 3, 1, 0, 6, 0, 0, 0, -1, -1, -1, -1, -1, -1],
            [2, 1, 0, 4, 0, 4, 3, 105, 0.5, 0.5, 0.3, 0.4, 0.4, 0.4],
            [2, 1, 0, 4, 0, 4, 3, 105, 0.45, 0.45, 0.33, 0.4, 0.4, 0.4],
            [0, 0, 4, 3, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1],
        ],
    ],
)
def test__sample_actions_backward__returns_valid_actions(env, states):
    """
    Just a little higher...
    """
    n_states = len(states)
    # Get masks
    masks = tbool(
        [env.get_mask_invalid_actions_backward(s) for s in states], device=env.device
    )
    # Build policy outputs
    params = env.random_distr_params
    policy_outputs = torch.tile(env.get_policy_output(params), dims=(n_states, 1))
    # Sample actions
    actions, _ = env.sample_actions_batch(
        policy_outputs, masks, states, is_backward=True
    )
    # Sample actions are valid
    for state, action in zip(states, actions):
        if env._get_stage(state) == Stage.LATTICE_PARAMETERS:
            continue
        assert action in env.get_valid_actions(state, done=False, backward=True)


@pytest.mark.repeat(100)
def test__trajectory_random__does_not_crash_from_source(env):
    """
    Raising the bar...
    """
    env.reset()
    env.trajectory_random()
    pass


@pytest.mark.parametrize(
    "states, actions",
    [
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
                (1, 7, -2, -2, -2, -2, -2),
                (3, 16, -2, -2, -2, -2, -2),
                (1, 6, -2, -2, -2, -2, -2),
                (3, 8, -2, -2, -2, -2, -2),
                (2, 11, -2, -2, -2, -2, -2),
                (3, 9, -2, -2, -2, -2, -2),
            ],
        ],
        [
            [
                [0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1],
                [1, 3, 1, 0, 6, 0, 0, 0, -1, -1, -1, -1, -1, -1],
                [1, 3, 1, 0, 6, 1, 0, 0, -1, -1, -1, -1, -1, -1],
                [1, 3, 1, 0, 6, 1, 1, 0, -1, -1, -1, -1, -1, -1],
            ],
            [
                (1, 6, -2, -2, -2, -2, -2),
                (2, 14, 0, -3, -3, -3, -3),
                (2, 2, 1, -3, -3, -3, -3),
                (2, 1, 3, -3, -3, -3, -3),
            ],
        ],
        [
            [
                [0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1],
                [0, 0, 4, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1],
                [1, 3, 1, 0, 6, 0, 0, 0, -1, -1, -1, -1, -1, -1],
                [2, 1, 0, 4, 0, 4, 3, 105, 0.1, 0.1, 0.3, 0.4, 0.4, 0.4],
                [1, 3, 1, 0, 6, 1, 0, 0, -1, -1, -1, -1, -1, -1],
                [1, 3, 1, 0, 6, 1, 1, 0, -1, -1, -1, -1, -1, -1],
                [0, 3, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1],
                [0, 3, 0, 0, 6, 0, 0, 0, -1, -1, -1, -1, -1, -1],
                [1, 3, 1, 0, 6, 1, 2, 0, -1, -1, -1, -1, -1, -1],
                [0, 3, 1, 0, 6, 0, 0, 0, -1, -1, -1, -1, -1, -1],
                [2, 1, 0, 4, 0, 4, 3, 105, 0.5, 0.5, 0.3, 0.4, 0.4, 0.4],
                [2, 1, 0, 4, 0, 4, 3, 105, 0.45, 0.45, 0.33, 0.4, 0.4, 0.4],
                [0, 0, 4, 3, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1],
            ],
            [
                (1, 15, -2, -2, -2, -2, -2),
                (1, 2, -2, -2, -2, -2, -2),
                (2, 7, 0, -3, -3, -3, -3),
                (0.49, 0.40, 0.40, 0.37, 0.35, 0.36, 0.0),
                (2, 1, 1, -3, -3, -3, -3),
                (2, 1, 3, -3, -3, -3, -3),
                (2, 11, -2, -2, -2, -2, -2),
                (3, 9, -2, -2, -2, -2, -2),
                (2, 2, 3, -3, -3, -3, -3),
                (3, 2, -2, -2, -2, -2, -2),
                (0.27, 0.28, 0.30, 0.39, 0.37, 0.29, 0.0),
                (0.32, 0.30, 0.45, 0.33, 0.42, 0.39, 0.0),
                (4, 4, -2, -2, -2, -2, -2),
            ],
        ],
    ],
)
def test__get_logprobs_forward__returns_valid_actions(env, states, actions):
    """
    This would already be not too bad!
    """
    n_states = len(states)
    actions = tfloat(actions, float_type=env.float, device=env.device)
    # Get masks
    masks = tbool(
        [env.get_mask_invalid_actions_forward(s) for s in states], device=env.device
    )
    # Build policy outputs
    params = env.random_distr_params
    policy_outputs = torch.tile(env.get_policy_output(params), dims=(n_states, 1))
    # Get log probs
    logprobs = env.get_logprobs(
        policy_outputs, actions, masks, states, is_backward=False
    )
    assert torch.all(torch.isfinite(logprobs))


# TODO: Set lattice system
@pytest.mark.parametrize(
    "states, actions",
    [
        [
            [
                [0, 0, 4, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1],
                [0, 0, 4, 3, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1],
                [0, 3, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1],
                [0, 3, 0, 0, 6, 0, 0, 0, -1, -1, -1, -1, -1, -1],
                [0, 3, 1, 0, 6, 0, 0, 0, -1, -1, -1, -1, -1, -1],
            ],
            [
                (2, 4, -2, -2, -2, -2, -2),
                (2, 4, -2, -2, -2, -2, -2),
                (1, 3, -2, -2, -2, -2, -2),
                (1, 3, -2, -2, -2, -2, -2),
                (4, 6, -2, -2, -2, -2, -2),
            ],
        ],
        [
            [
                [1, 3, 1, 0, 6, 0, 0, 0, -1, -1, -1, -1, -1, -1],
                [1, 3, 1, 0, 6, 1, 0, 0, -1, -1, -1, -1, -1, -1],
                [1, 3, 1, 0, 6, 1, 1, 0, -1, -1, -1, -1, -1, -1],
            ],
            [
                (-1, -1, -2, -2, -2, -2, -2),
                (0, 1, 0, -3, -3, -3, -3),
                (1, 1, 1, -3, -3, -3, -3),
            ],
        ],
        [
            [
                [0, 0, 4, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1],
                [1, 3, 1, 0, 6, 0, 0, 0, -1, -1, -1, -1, -1, -1],
                #                 [2, 1, 0, 4, 0, 4, 3, 105, 0.1, 0.1, 0.3, 0.4, 0.4, 0.4],
                [1, 3, 1, 0, 6, 1, 0, 0, -1, -1, -1, -1, -1, -1],
                [1, 3, 1, 0, 6, 1, 1, 0, -1, -1, -1, -1, -1, -1],
                [0, 3, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1],
                [0, 3, 0, 0, 6, 0, 0, 0, -1, -1, -1, -1, -1, -1],
                [1, 3, 1, 0, 6, 1, 2, 0, -1, -1, -1, -1, -1, -1],
                [0, 3, 1, 0, 6, 0, 0, 0, -1, -1, -1, -1, -1, -1],
                #                 [2, 1, 0, 4, 0, 4, 3, 105, 0.5, 0.5, 0.3, 0.4, 0.4, 0.4],
                #                 [2, 1, 0, 4, 0, 4, 3, 105, 0.45, 0.45, 0.33, 0.4, 0.4, 0.4],
                [0, 0, 4, 3, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1],
            ],
            [
                (2, 4, -2, -2, -2, -2, -2),
                (-1, -1, -2, -2, -2, -2, -2),
                #                 (0.10, 0.10, 0.17, 0.0, 0.0, 0.0, 0.0),
                (0, 1, 0, -3, -3, -3, -3),
                (1, 1, 1, -3, -3, -3, -3),
                (1, 3, -2, -2, -2, -2, -2),
                (1, 3, -2, -2, -2, -2, -2),
                (1, 2, 1, -3, -3, -3, -3),
                (2, 1, -2, -2, -2, -2, -2),
                #                 (0.37, 0.37, 0.23, 0.0, 0.0, 0.0, 0.0),
                #                 (0.23, 0.23, 0.11, 0.0, 0.0, 0.0, 0.0),
                (3, 3, -2, -2, -2, -2, -2),
            ],
        ],
    ],
)
def test__get_logprobs_backward__returns_valid_actions(env, states, actions):
    """
    And backwards?
    """
    n_states = len(states)
    actions = tfloat(actions, float_type=env.float, device=env.device)
    # Get masks
    masks = tbool(
        [env.get_mask_invalid_actions_backward(s) for s in states], device=env.device
    )
    # Build policy outputs
    params = env.random_distr_params
    policy_outputs = torch.tile(env.get_policy_output(params), dims=(n_states, 1))
    # Get log probs
    logprobs = env.get_logprobs(
        policy_outputs, actions, masks, states, is_backward=True
    )
    assert torch.all(torch.isfinite(logprobs))


class TestContinuousCrystalBasic(common.BaseTestsContinuous):
    """Common tests for crystal without composition <-> space group constraints."""

    @pytest.fixture(autouse=True)
    def setup(self, env):
        self.env = env
        self.repeats = {
            "test__reset__state_is_source": 10,
        }
        self.n_states = {}  # TODO: Populate.


class TestContinuousCrystalSGCheck(common.BaseTestsContinuous):
    """Common tests for crystal with composition <-> space group constraints."""

    @pytest.fixture(autouse=True)
    def setup(self, env_with_stoichiometry_sg_check):
        self.env = env_with_stoichiometry_sg_check
        self.repeats = {
            "test__set_state__creates_new_copy_of_state": 10,  # Overrides no repeat.
            "test__reset__state_is_source": 0,
        }
        self.n_states = {}  # TODO: Populate.


class TestContinuousCrystalSGFirst(common.BaseTestsContinuous):
    """Common tests for crystal with space group first."""

    @pytest.fixture(autouse=True)
    def setup(self, env_sg_first):
        self.env = env_sg_first
        self.repeats = {
            "test__get_logprobs__backward__returns_zero_if_done": 100,  # Overrides no repeat.
            "test__reset__state_is_source": 10,
        }
        self.n_states = {}  # TODO: Populate.
