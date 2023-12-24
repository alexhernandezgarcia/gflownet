import warnings

import common
import numpy as np
import pytest
import torch
from torch import Tensor

from gflownet.envs.crystals.catalyst import Catalyst, Stage
from gflownet.utils.common import tbool, tfloat
from gflownet.utils.crystals.constants import HEXAGONAL, RHOMBOHEDRAL, TRICLINIC

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
    143,  # hexagonal
    144,  # hexagonal
    146,  # rhombohedral
    148,  # rhombohedral
    168,  # hexagonal
    169,  # hexagonal
    189,  # hexagonal
    195,
    200,
    230,
]


@pytest.fixture
def env_no_sg_to_miller():
    return Catalyst(
        composition_kwargs={"elements": 4},
        space_group_kwargs={"space_groups_subset": SG_SUBSET_ALL_CLS_PS},
        do_sg_before_composition=True,
        do_sg_to_composition_constraints=True,
        do_sg_to_lp_constraints=True,
        do_sg_to_miller_constraints=False,
    )


@pytest.fixture
def env_sg_to_miller():
    return Catalyst(
        composition_kwargs={"elements": 4},
        space_group_kwargs={"space_groups_subset": SG_SUBSET_ALL_CLS_PS},
        do_sg_before_composition=True,
        do_sg_to_composition_constraints=True,
        do_sg_to_lp_constraints=True,
        do_sg_to_miller_constraints=True,
    )


@pytest.mark.parametrize("env", ["env_no_sg_to_miller", "env_sg_to_miller"])
def test__environment__initializes_properly(env, request):
    env = request.getfixturevalue(env)
    assert True


@pytest.mark.parametrize("env", ["env_no_sg_to_miller", "env_sg_to_miller"])
def test__stage_next__returns_expected(env, request):
    env = request.getfixturevalue(env)
    assert env._get_next_stage(None) == Stage.SPACE_GROUP
    assert env._get_next_stage(Stage.SPACE_GROUP) == Stage.COMPOSITION
    assert env._get_next_stage(Stage.COMPOSITION) == Stage.LATTICE_PARAMETERS
    assert env._get_next_stage(Stage.LATTICE_PARAMETERS) == Stage.MILLER_INDICES
    assert env._get_next_stage(Stage.MILLER_INDICES) == Stage.DONE
    assert env._get_next_stage(Stage.DONE) == None


@pytest.mark.parametrize("env", ["env_no_sg_to_miller", "env_sg_to_miller"])
def test__stage_prev__returns_expected(env, request):
    env = request.getfixturevalue(env)
    assert env._get_previous_stage(Stage.SPACE_GROUP) == Stage.DONE
    assert env._get_previous_stage(Stage.COMPOSITION) == Stage.SPACE_GROUP
    assert env._get_previous_stage(Stage.LATTICE_PARAMETERS) == Stage.COMPOSITION
    assert env._get_previous_stage(Stage.MILLER_INDICES) == Stage.LATTICE_PARAMETERS
    assert env._get_previous_stage(Stage.DONE) == Stage.MILLER_INDICES


@pytest.mark.parametrize("env", ["env_no_sg_to_miller", "env_sg_to_miller"])
def test__environment__has_expected_initial_state(env, request):
    """
    The source of the composition, space group and Miller indices environments is all
    0s. The source of the continuous lattice parameters environment is all -1s.
    """
    env = request.getfixturevalue(env)
    expected_initial_state = [1] + [0] * (4 + 3) + [-1] * 6 + [0] * 3
    assert (
        env.state == env.source == expected_initial_state
    )  # stage + n elements + space groups + lattice parameters + 3 miller indices


@pytest.mark.parametrize("env", ["env_no_sg_to_miller", "env_sg_to_miller"])
def test__environment__has_expected_action_space(env, request):
    env = request.getfixturevalue(env)
    assert len(env.action_space) == len(
        env.subenvs[Stage.COMPOSITION].action_space
    ) + len(env.subenvs[Stage.SPACE_GROUP].action_space) + len(
        env.subenvs[Stage.LATTICE_PARAMETERS].action_space
    ) + len(
        env.subenvs[Stage.MILLER_INDICES].action_space
    )

    underlying_action_space = (
        env.subenvs[Stage.COMPOSITION].action_space
        + env.subenvs[Stage.SPACE_GROUP].action_space
        + env.subenvs[Stage.LATTICE_PARAMETERS].action_space
        + env.subenvs[Stage.MILLER_INDICES].action_space
    )

    for action, underlying_action in zip(env.action_space, underlying_action_space):
        assert action[: len(underlying_action)] == underlying_action


@pytest.mark.parametrize("env", ["env_no_sg_to_miller", "env_sg_to_miller"])
def test__pad_depad_action(env, request):
    env = request.getfixturevalue(env)
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
            [0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            [0, 0, 4, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            [1, 3, 1, 0, 6, 0, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            [2, 1, 0, 4, 0, 1, 2, 2, 0.12, 0.23, 0.34, 0.45, 0.56, 0.67, 0, 0, 0],
            [3, 1, 0, 4, 0, 1, 2, 2, 0.12, 0.23, 0.34, 0.45, 0.56, 0.67, 1, 2, 3],
            [1, 3, 1, 0, 6, 1, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            [1, 3, 1, 0, 6, 1, 1, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            [0, 3, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            [0, 3, 0, 0, 6, 0, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            [1, 3, 1, 0, 6, 1, 2, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            [0, 3, 1, 0, 6, 0, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            [2, 1, 0, 4, 0, 1, 2, 2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0, 0, 0],
            [2, 1, 0, 4, 0, 1, 2, 2, 0.76, 0.75, 0.74, 0.73, 0.72, 0.71, 0, 0, 0],
            [3, 1, 0, 4, 0, 1, 2, 2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0, 4, 2],
            [3, 1, 0, 4, 0, 1, 2, 2, 0.76, 0.75, 0.74, 0.73, 0.72, 0.71, 1, 3, 0],
            [0, 0, 4, 3, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0],
        ],
    ],
)
def test__states2policy__is_concatenation_of_subenv_states(env_no_sg_to_miller, states):
    env = env_no_sg_to_miller
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
            [0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            [0, 0, 4, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            [1, 3, 1, 0, 6, 0, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            [2, 1, 0, 4, 0, 1, 2, 2, 0.12, 0.23, 0.34, 0.45, 0.56, 0.67, 0, 0, 0],
            [3, 1, 0, 4, 0, 1, 2, 2, 0.12, 0.23, 0.34, 0.45, 0.56, 0.67, 1, 2, 3],
            [1, 3, 1, 0, 6, 1, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            [1, 3, 1, 0, 6, 1, 1, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            [0, 3, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            [0, 3, 0, 0, 6, 0, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            [1, 3, 1, 0, 6, 1, 2, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            [0, 3, 1, 0, 6, 0, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            [2, 1, 0, 4, 0, 1, 2, 2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0, 0, 0],
            [2, 1, 0, 4, 0, 1, 2, 2, 0.76, 0.75, 0.74, 0.73, 0.72, 0.71, 0, 0, 0],
            [3, 1, 0, 4, 0, 1, 2, 2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0, 4, 2],
            [3, 1, 0, 4, 0, 1, 2, 2, 0.76, 0.75, 0.74, 0.73, 0.72, 0.71, 1, 3, 0],
            [0, 0, 4, 3, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0],
        ],
    ],
)
def test__states2proxy__is_concatenation_of_subenv_states(env_no_sg_to_miller, states):
    env = env_no_sg_to_miller
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
            [0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            [0, 0, 4, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            [1, 3, 1, 0, 6, 0, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            [2, 1, 0, 4, 0, 1, 2, 2, 0.12, 0.23, 0.34, 0.45, 0.56, 0.67, 0, 0, 0],
            [3, 1, 0, 4, 0, 1, 2, 2, 0.12, 0.23, 0.34, 0.45, 0.56, 0.67, 1, 2, 3],
            [1, 3, 1, 0, 6, 1, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            [1, 3, 1, 0, 6, 1, 1, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            [0, 3, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            [0, 3, 0, 0, 6, 0, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            [1, 3, 1, 0, 6, 1, 2, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            [0, 3, 1, 0, 6, 0, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            [2, 1, 0, 4, 0, 1, 2, 2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0, 0, 0],
            [2, 1, 0, 4, 0, 1, 2, 2, 0.76, 0.75, 0.74, 0.73, 0.72, 0.71, 0, 0, 0],
            [3, 1, 0, 4, 0, 1, 2, 2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0, 4, 2],
            [3, 1, 0, 4, 0, 1, 2, 2, 0.76, 0.75, 0.74, 0.73, 0.72, 0.71, 1, 3, 0],
            [0, 0, 4, 3, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0],
        ],
    ],
)
def test__state2readable__is_concatenation_of_subenv_states(
    env_no_sg_to_miller, states
):
    env = env_no_sg_to_miller
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
            f"LatticeParameters = {readables[2]};"
            f"MillerIndices = {readables[3]}"
        )
    # Get policy states from env.states2policy
    states_readable = [env.state2readable(state) for state in states]
    for readable, readable_expected in zip(states_readable, states_readable_expected):
        assert readable == readable_expected


@pytest.mark.parametrize(
    "state, state_composition, state_space_group, state_lattice_parameters, state_miller",
    [
        [
            [1, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0],
            [-1, -1, -1, -1, -1, -1],
            [0, 0, 0],
        ],
        [
            [1, 0, 0, 0, 0, 1, 2, 2, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 2, 2],
            [-1, -1, -1, -1, -1, -1],
            [0, 0, 0],
        ],
        [
            [0, 0, 0, 0, 0, 1, 2, 2, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 2, 2],
            [-1, -1, -1, -1, -1, -1],
            [0, 0, 0],
        ],
        [
            [0, 1, 0, 4, 0, 1, 2, 2, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            [1, 0, 4, 0],
            [1, 2, 2],
            [-1, -1, -1, -1, -1, -1],
            [0, 0, 0],
        ],
        [
            [2, 1, 0, 4, 0, 1, 2, 2, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            [1, 0, 4, 0],
            [1, 2, 2],
            [-1, -1, -1, -1, -1, -1],
            [0, 0, 0],
        ],
        [
            [2, 1, 0, 4, 0, 1, 2, 2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0, 0, 0],
            [1, 0, 4, 0],
            [1, 2, 2],
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            [0, 0, 0],
        ],
        [
            [3, 1, 0, 4, 0, 1, 2, 2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1, 2, 3],
            [1, 0, 4, 0],
            [1, 2, 2],
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            [1, 2, 3],
        ],
        [
            [3, 1, 0, 4, 0, 1, 2, 2, 0.76, 0.75, 0.74, 0.73, 0.72, 0.71, 1, 3, 0],
            [1, 0, 4, 0],
            [1, 2, 2],
            [0.76, 0.75, 0.74, 0.73, 0.72, 0.71],
            [1, 3, 0],
        ],
        [
            [3, 1, 0, 4, 0, 1, 2, 2, 0.76, 0.75, 0.74, 0.73, 0.72, 0.71, 0, 4, 2],
            [1, 0, 4, 0],
            [1, 2, 2],
            [0.76, 0.75, 0.74, 0.73, 0.72, 0.71],
            [0, 4, 2],
        ],
    ],
)
def test__state_of_subenv__returns_expected(
    env_no_sg_to_miller,
    state,
    state_composition,
    state_space_group,
    state_lattice_parameters,
    state_miller,
):
    env = env_no_sg_to_miller
    for stage in env.subenvs:
        state_subenv = env._get_state_of_subenv(state, stage)
        if stage == Stage.COMPOSITION:
            assert state_subenv == state_composition
        elif stage == Stage.SPACE_GROUP:
            assert state_subenv == state_space_group
        elif stage == Stage.LATTICE_PARAMETERS:
            assert state_subenv == state_lattice_parameters
        elif stage == Stage.MILLER_INDICES:
            assert state_subenv == state_miller
        else:
            raise ValueError(f"Unrecognized stage {stage}.")


@pytest.mark.parametrize(
    "env, state, dones, has_sg_to_lp_constraints, has_sg_to_miller_constraints, "
    "has_composition_to_sg_constraints, has_sg_to_composition_constraints",
    [
        (
            "env_no_sg_to_miller",
            [1, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            [False, False, False, False],
            False,
            False,
            False,
            False,
        ),
        (
            "env_no_sg_to_miller",
            [1, 0, 0, 0, 0, 1, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            [False, False, False, False],
            True,
            False,
            False,
            False,
        ),
        (
            "env_sg_to_miller",
            [1, 0, 0, 0, 0, 1, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            [False, False, False, False],
            True,
            True,
            False,
            False,
        ),
        (
            "env_sg_to_miller",
            [0, 0, 0, 0, 0, 1, 2, 2, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            [False, True, False, False],
            True,
            True,
            False,
            True,
        ),
        (
            "env_sg_to_miller",
            [0, 1, 0, 4, 0, 1, 2, 2, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            [False, True, False, False],
            True,
            True,
            False,
            True,
        ),
        (
            "env_sg_to_miller",
            [2, 1, 0, 4, 0, 1, 2, 2, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            [True, True, False, False],
            True,
            True,
            False,
            True,
        ),
        (
            "env_sg_to_miller",
            [2, 1, 0, 4, 0, 1, 2, 2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0, 0, 0],
            [True, True, False, False],
            True,
            True,
            False,
            True,
        ),
        (
            "env_sg_to_miller",
            [3, 1, 0, 4, 0, 1, 2, 2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0, 0, 0],
            [True, True, True, False],
            True,
            True,
            False,
            True,
        ),
        (
            "env_sg_to_miller",
            [3, 1, 0, 4, 0, 1, 2, 2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0, 1, 3],
            [True, True, True, False],
            True,
            True,
            False,
            True,
        ),
        (
            "env_sg_to_miller",
            [2, 1, 0, 4, 0, 6, 1, 143, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            [True, True, False, False],
            True,
            True,
            False,
            False,
        ),
        (
            "env_sg_to_miller",
            [2, 1, 0, 4, 0, 5, 1, 146, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            [True, True, False, False],
            True,
            True,
            False,
            False,
        ),
    ],
)
def test__set_state__sets_state_subenvs_dones_and_constraints(
    env,
    state,
    dones,
    has_sg_to_lp_constraints,
    has_sg_to_miller_constraints,
    has_composition_to_sg_constraints,
    has_sg_to_composition_constraints,
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
        assert subenv.done == done

    # Check lattice system constraints
    if env.subenvs[Stage.SPACE_GROUP].lattice_system != "None":
        # SG to LP constraints
        if has_sg_to_lp_constraints:
            assert (
                env.subenvs[Stage.SPACE_GROUP].lattice_system
                == env.subenvs[Stage.LATTICE_PARAMETERS].lattice_system
            )
        else:
            assert env.subenvs[Stage.LATTICE_PARAMETERS].lattice_system == TRICLINIC
        # SG to Miller constraints
        if has_sg_to_miller_constraints:
            assert (
                env.subenvs[Stage.SPACE_GROUP].lattice_system
                in [HEXAGONAL, RHOMBOHEDRAL]
            ) == env.subenvs[Stage.MILLER_INDICES].is_hexagonal_rhombohedral
        else:
            assert env.subenvs[Stage.MILLER_INDICES].is_hexagonal_rhombohedral is False

    # Check composition constraints
    if has_composition_to_sg_constraints:
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
    if has_sg_to_composition_constraints:
        assert (
            env.subenvs[Stage.COMPOSITION].space_group
            == env.subenvs[Stage.SPACE_GROUP].space_group
        )


@pytest.mark.parametrize(
    "env, state",
    [
        ("env_sg_to_miller", [1, 0, 0, 0, 0, 1, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0]),
        ("env_sg_to_miller", [1, 0, 0, 0, 0, 1, 2, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0]),
        ("env_sg_to_miller", [1, 0, 0, 0, 0, 1, 2, 2, -1, -1, -1, -1, -1, -1, 0, 0, 0]),
        ("env_sg_to_miller", [1, 0, 0, 0, 0, 6, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0]),
        ("env_sg_to_miller", [1, 0, 0, 0, 0, 6, 1, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0]),
        (
            "env_sg_to_miller",
            [1, 0, 0, 0, 0, 6, 1, 143, -1, -1, -1, -1, -1, -1, 0, 0, 0],
        ),
        ("env_sg_to_miller", [0, 1, 0, 4, 0, 1, 2, 2, -1, -1, -1, -1, -1, -1, 0, 0, 0]),
        (
            "env_sg_to_miller",
            [2, 1, 0, 4, 0, 1, 2, 2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0, 0, 0],
        ),
        (
            "env_sg_to_miller",
            [3, 1, 0, 4, 0, 1, 2, 2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0, 1, 3],
        ),
    ],
)
def test__get_mask_invalid_actions_backward__returns_expected_general_case(
    env, state, request
):
    env = request.getfixturevalue(env)
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
    "env, state",
    [
        ("env_sg_to_miller", [0, 0, 0, 0, 0, 1, 2, 2, -1, -1, -1, -1, -1, -1, 0, 0, 0]),
        ("env_sg_to_miller", [0, 0, 0, 0, 0, 1, 2, 2, -1, -1, -1, -1, -1, -1, 0, 0, 0]),
        (
            "env_sg_to_miller",
            [2, 1, 0, 4, 0, 6, 1, 143, -1, -1, -1, -1, -1, -1, 0, 0, 0],
        ),
        (
            "env_sg_to_miller",
            [3, 1, 0, 4, 0, 1, 2, 2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0, 0, 0],
        ),
    ],
)
def test__get_mask_invald_actions_backward__returns_expected_stage_transition(
    env, state, request
):
    env = request.getfixturevalue(env)
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


@pytest.mark.repeat(10)
@pytest.mark.parametrize("env", ["env_no_sg_to_miller", "env_sg_to_miller"])
def test__step_random__does_not_crash_from_source(env, request):
    """
    Very low bar test...
    """
    env = request.getfixturevalue(env)
    env.reset()
    env.step_random()
    assert True


@pytest.mark.parametrize(
    "env, actions, exp_result, exp_stage, last_action_valid",
    [
        [
            "env_sg_to_miller",
            [(0, 1, 0, -3, -3, -3, -3)],
            [1, 0, 0, 0, 0, 1, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            Stage.SPACE_GROUP,
            True,
        ],
        [
            "env_sg_to_miller",
            [(0, 1, 0, -3, -3, -3, -3), (1, 2, 1, -3, -3, -3, -3)],
            [1, 0, 0, 0, 0, 1, 2, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            Stage.SPACE_GROUP,
            True,
        ],
        [
            "env_sg_to_miller",
            [
                (0, 1, 0, -3, -3, -3, -3),
                (1, 2, 1, -3, -3, -3, -3),
                (2, 2, 3, -3, -3, -3, -3),
            ],
            [1, 0, 0, 0, 0, 1, 2, 2, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            Stage.SPACE_GROUP,
            True,
        ],
        [
            "env_sg_to_miller",
            [
                (0, 6, 0, -3, -3, -3, -3),
                (1, 1, 1, -3, -3, -3, -3),
                (2, 143, 3, -3, -3, -3, -3),
            ],
            [1, 0, 0, 0, 0, 6, 1, 143, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            Stage.SPACE_GROUP,
            True,
        ],
        [
            "env_sg_to_miller",
            [
                (0, 6, 0, -3, -3, -3, -3),
                (1, 1, 1, -3, -3, -3, -3),
                (2, 143, 3, -3, -3, -3, -3),
                (-1, -1, -1, -3, -3, -3, -3),
                (1, 1, -2, -2, -2, -2, -2),
                (3, 4, -2, -2, -2, -2, -2),
            ],
            [0, 1, 0, 4, 0, 6, 1, 143, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            Stage.COMPOSITION,
            True,
        ],
        [
            "env_sg_to_miller",
            [
                (0, 6, 0, -3, -3, -3, -3),
                (1, 1, 1, -3, -3, -3, -3),
                (2, 143, 3, -3, -3, -3, -3),
                (-1, -1, -1, -3, -3, -3, -3),
                (1, 1, -2, -2, -2, -2, -2),
                (3, 4, -2, -2, -2, -2, -2),
                (-1, -1, -2, -2, -2, -2, -2),
            ],
            [2, 1, 0, 4, 0, 6, 1, 143, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            Stage.LATTICE_PARAMETERS,
            True,
        ],
        [
            "env_sg_to_miller",
            [
                (0, 6, 0, -3, -3, -3, -3),
                (1, 1, 1, -3, -3, -3, -3),
                (2, 143, 3, -3, -3, -3, -3),
                (-1, -1, -1, -3, -3, -3, -3),
                (1, 1, -2, -2, -2, -2, -2),
                (3, 4, -2, -2, -2, -2, -2),
                (-1, -1, -2, -2, -2, -2, -2),
                (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1),
            ],
            [2, 1, 0, 4, 0, 6, 1, 143, 0.1, 0.1, 0.3, 0.4, 0.4, 0.7, 0, 0, 0],
            Stage.LATTICE_PARAMETERS,
            True,
        ],
        [
            "env_sg_to_miller",
            [
                (0, 6, 0, -3, -3, -3, -3),
                (1, 1, 1, -3, -3, -3, -3),
                (2, 143, 3, -3, -3, -3, -3),
                (-1, -1, -1, -3, -3, -3, -3),
                (1, 1, -2, -2, -2, -2, -2),
                (3, 4, -2, -2, -2, -2, -2),
                (-1, -1, -2, -2, -2, -2, -2),
                (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1),
                (0.66, 0.0, 0.44, 0.0, 0.0, 0.0, 0),
            ],
            [2, 1, 0, 4, 0, 6, 1, 143, 0.76, 0.76, 0.74, 0.4, 0.4, 0.7, 0, 0, 0],
            Stage.LATTICE_PARAMETERS,
            True,
        ],
        [
            "env_sg_to_miller",
            [
                (0, 6, 0, -3, -3, -3, -3),
                (1, 1, 1, -3, -3, -3, -3),
                (2, 143, 3, -3, -3, -3, -3),
                (-1, -1, -1, -3, -3, -3, -3),
                (1, 1, -2, -2, -2, -2, -2),
                (3, 4, -2, -2, -2, -2, -2),
                (-1, -1, -2, -2, -2, -2, -2),
                (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1),
                (0.66, 0.0, 0.44, 0.0, 0.0, 0.0, 0),
                (np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf),
            ],
            [3, 1, 0, 4, 0, 6, 1, 143, 0.76, 0.76, 0.74, 0.4, 0.4, 0.7, 0, 0, 0],
            Stage.MILLER_INDICES,
            True,
        ],
        [
            "env_sg_to_miller",
            [
                (0, 6, 0, -3, -3, -3, -3),
                (1, 1, 1, -3, -3, -3, -3),
                (2, 143, 3, -3, -3, -3, -3),
                (-1, -1, -1, -3, -3, -3, -3),
                (1, 1, -2, -2, -2, -2, -2),
                (3, 4, -2, -2, -2, -2, -2),
                (-1, -1, -2, -2, -2, -2, -2),
                (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1),
                (0.66, 0.0, 0.44, 0.0, 0.0, 0.0, 0),
                (np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf),
                (1, 0, 0, -5, -5, -5, -5),
            ],
            [3, 1, 0, 4, 0, 6, 1, 143, 0.76, 0.76, 0.74, 0.4, 0.4, 0.7, 1, 0, 0],
            Stage.MILLER_INDICES,
            True,
        ],
        [
            "env_sg_to_miller",
            [
                (2, 143, 0, -3, -3, -3, -3),
                (-1, -1, -1, -3, -3, -3, -3),
                (1, 1, -2, -2, -2, -2, -2),
                (3, 4, -2, -2, -2, -2, -2),
                (-1, -1, -2, -2, -2, -2, -2),
                (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1),
                (np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf),
                (1, 0, 0, -5, -5, -5, -5),
            ],
            [3, 1, 0, 4, 0, 6, 1, 143, 0.1, 0.1, 0.3, 0.4, 0.4, 0.7, 1, 0, 0],
            Stage.MILLER_INDICES,
            True,
        ],
    ],
)
def test__step__action_sequence_has_expected_result(
    env, actions, exp_result, exp_stage, last_action_valid, request
):
    env = request.getfixturevalue(env)
    for action in actions:
        warnings.filterwarnings("ignore")
        _, _, valid = env.step(action)

    assert env.state == exp_result
    assert env._get_stage() == exp_stage
    assert valid == last_action_valid


@pytest.mark.parametrize(
    "env, state_init, state_end, stage_init, stage_end, actions, last_action_valid",
    [
        [
            "env_sg_to_miller",
            [1, 0, 0, 0, 0, 1, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            Stage.SPACE_GROUP,
            Stage.SPACE_GROUP,
            [(0, 1, 0, -3, -3, -3, -3)],
            True,
        ],
        [
            "env_sg_to_miller",
            [1, 0, 0, 0, 0, 1, 2, 2, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            Stage.SPACE_GROUP,
            Stage.SPACE_GROUP,
            [
                (2, 2, 3, -3, -3, -3, -3),
                (1, 2, 1, -3, -3, -3, -3),
                (0, 1, 0, -3, -3, -3, -3),
            ],
            True,
        ],
        [
            "env_sg_to_miller",
            [3, 1, 0, 4, 0, 6, 1, 143, 0.76, 0.76, 0.74, 0.4, 0.4, 0.7, 1, 0, 0],
            [3, 1, 0, 4, 0, 6, 1, 143, 0.76, 0.76, 0.74, 0.4, 0.4, 0.7, 0, 0, 0],
            Stage.MILLER_INDICES,
            Stage.MILLER_INDICES,
            [
                (1, 0, 0, -5, -5, -5, -5),
            ],
            True,
        ],
        [
            "env_sg_to_miller",
            [3, 1, 0, 4, 0, 6, 1, 143, 0.76, 0.76, 0.74, 0.4, 0.4, 0.7, 1, 0, 0],
            [2, 1, 0, 4, 0, 6, 1, 143, 0.76, 0.76, 0.74, 0.4, 0.4, 0.7, 0, 0, 0],
            Stage.MILLER_INDICES,
            Stage.LATTICE_PARAMETERS,
            [
                (1, 0, 0, -5, -5, -5, -5),
                (np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf),
            ],
            True,
        ],
        [
            "env_sg_to_miller",
            [3, 1, 0, 4, 0, 6, 1, 143, 0.76, 0.76, 0.74, 0.4, 0.4, 0.7, 1, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            Stage.MILLER_INDICES,
            Stage.SPACE_GROUP,
            [
                (1, 0, 0, -5, -5, -5, -5),
                (np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf),
                (0.66, 0.0, 0.44, 0.0, 0.0, 0.0, 0),
                (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1),
                (-1, -1, -2, -2, -2, -2, -2),
                (3, 4, -2, -2, -2, -2, -2),
                (1, 1, -2, -2, -2, -2, -2),
                (-1, -1, -1, -3, -3, -3, -3),
                (2, 143, 3, -3, -3, -3, -3),
                (1, 1, 1, -3, -3, -3, -3),
                (0, 6, 0, -3, -3, -3, -3),
            ],
            True,
        ],
    ],
)
def test__step_backwards__action_sequence_has_expected_result(
    env,
    state_init,
    state_end,
    stage_init,
    stage_end,
    actions,
    last_action_valid,
    request,
):
    env = request.getfixturevalue(env)

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


@pytest.mark.skip(
    reason="This is not the correct way of checking an exception is raised"
)
@pytest.mark.parametrize(
    "env, state, action",
    [
        [
            "env_sg_to_miller",
            [1, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            (1, 1, -2, -2, -2, -2, -2),
        ],
    ],
)
def test__step__actions_not_in_subenv_raise_exception(env, state, action, request):
    env = request.getfixturevalue(env)
    env.set_state(state)
    stage = env._get_stage(env.state)
    subenv = env.subenvs[stage]
    action_subenv = env._depad_action(action, stage)
    action_to_check = subenv.action2representative(action_subenv)
    with pytest.raises(
        ValueError,
        match=f"Tried to execute action {action_to_check} not present in action space.",
    ):
        _, _, valid = env.step(action)


@pytest.mark.repeat(100)
@pytest.mark.parametrize("env", ["env_no_sg_to_miller", "env_sg_to_miller"])
def test__trajectory_random__does_not_crash_from_source(env, request):
    """
    Raising the bar...
    """
    env = request.getfixturevalue(env)
    env.reset()
    env.trajectory_random()
    assert True


@pytest.mark.skip(reason="skip while developping other tests")
def test__continuous_env_common(env_sg_to_miller):
    print("\n\nCommon tests for catalyst with constraints from SG to Miller indices\n")
    return common.test__continuous_env_common(env_sg_to_miller)
