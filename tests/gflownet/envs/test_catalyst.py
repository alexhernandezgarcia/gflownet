import warnings

import common
import numpy as np
import pytest
import torch
from torch import Tensor

from gflownet.envs.crystals.catalyst import Catalyst, Stage
from gflownet.utils.common import tbool, tfloat


@pytest.fixture
def env():
    return Catalyst(
        composition_kwargs={"elements": 4},
        do_composition_to_sg_constraints=False,
        space_group_kwargs={"space_groups_subset": list(range(1, 15 + 1)) + [105]},
    )


@pytest.fixture
def env_sg_first():
    return Catalyst(
        composition_kwargs={"elements": 4},
        do_sg_to_composition_constraints=True,
        do_sg_before_composition=True,
    )


def test__stage_next__returns_expected(env, env_sg_first):
    assert env._get_next_stage(None) == Stage.COMPOSITION
    assert env._get_next_stage(Stage.COMPOSITION) == Stage.SPACE_GROUP
    assert env._get_next_stage(Stage.SPACE_GROUP) == Stage.LATTICE_PARAMETERS
    assert env._get_next_stage(Stage.LATTICE_PARAMETERS) == Stage.MILLER_INDICES
    assert env._get_next_stage(Stage.MILLER_INDICES) == Stage.DONE
    assert env._get_next_stage(Stage.DONE) == None

    assert env_sg_first._get_next_stage(None) == Stage.SPACE_GROUP
    assert env_sg_first._get_next_stage(Stage.SPACE_GROUP) == Stage.COMPOSITION
    assert env_sg_first._get_next_stage(Stage.COMPOSITION) == Stage.LATTICE_PARAMETERS
    assert env._get_next_stage(Stage.LATTICE_PARAMETERS) == Stage.MILLER_INDICES
    assert env._get_next_stage(Stage.MILLER_INDICES) == Stage.DONE
    assert env_sg_first._get_next_stage(Stage.DONE) == None


def test__stage_prev__returns_expected(env, env_sg_first):
    assert env._get_previous_stage(Stage.COMPOSITION) == Stage.DONE
    assert env._get_previous_stage(Stage.SPACE_GROUP) == Stage.COMPOSITION
    assert env._get_previous_stage(Stage.LATTICE_PARAMETERS) == Stage.SPACE_GROUP
    assert env._get_previous_stage(Stage.MILLER_INDICES) == Stage.LATTICE_PARAMETERS
    assert env._get_previous_stage(Stage.DONE) == Stage.MILLER_INDICES

    assert env_sg_first._get_previous_stage(Stage.SPACE_GROUP) == Stage.DONE
    assert env_sg_first._get_previous_stage(Stage.COMPOSITION) == Stage.SPACE_GROUP
    assert (
        env_sg_first._get_previous_stage(Stage.LATTICE_PARAMETERS) == Stage.COMPOSITION
    )
    assert (
        env_sg_first._get_previous_stage(Stage.MILLER_INDICES)
        == Stage.LATTICE_PARAMETERS
    )
    assert env_sg_first._get_previous_stage(Stage.DONE) == Stage.MILLER_INDICES


def test__environment__initializes_properly(env):
    pass


@pytest.mark.parametrize("env_input, initial_stage", [["env", 0], ["env_sg_first", 1]])
def test__environment__has_expected_initial_state(env_input, initial_stage, request):
    """
    The source of the composition, space group and Miller indices environments is all
    0s. The source of the continuous lattice parameters environment is all -1s.
    """
    env = request.getfixturevalue(env_input)
    expected_initial_state = [initial_stage] + [0] * (4 + 3) + [-1] * 6 + [0] * 3
    assert (
        env.state == env.source == expected_initial_state
    )  # stage + n elements + space groups + lattice parameters + 3 miller indices


def test__environment__has_expected_action_space(env):
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
            [0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            [0, 0, 4, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            [1, 3, 1, 0, 6, 0, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            [2, 1, 0, 4, 0, 4, 3, 105, 0.12, 0.23, 0.34, 0.45, 0.56, 0.67, 0, 0, 0],
            [3, 1, 0, 4, 0, 4, 3, 105, 0.12, 0.23, 0.34, 0.45, 0.56, 0.67, 1, 2, 3],
            [1, 3, 1, 0, 6, 1, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            [1, 3, 1, 0, 6, 1, 1, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            [0, 3, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            [0, 3, 0, 0, 6, 0, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            [1, 3, 1, 0, 6, 1, 2, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            [0, 3, 1, 0, 6, 0, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            [2, 1, 0, 4, 0, 4, 3, 105, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0, 0, 0],
            [2, 1, 0, 4, 0, 4, 3, 105, 0.76, 0.75, 0.74, 0.73, 0.72, 0.71, 0, 0, 0],
            [3, 1, 0, 4, 0, 4, 3, 105, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0, 4, 2],
            [3, 1, 0, 4, 0, 4, 3, 105, 0.76, 0.75, 0.74, 0.73, 0.72, 0.71, 1, 3, 0],
            [0, 0, 4, 3, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0],
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
            [0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            [0, 0, 4, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            [1, 3, 1, 0, 6, 0, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            [2, 1, 0, 4, 0, 4, 3, 105, 0.12, 0.23, 0.34, 0.45, 0.56, 0.67, 0, 0, 0],
            [3, 1, 0, 4, 0, 4, 3, 105, 0.12, 0.23, 0.34, 0.45, 0.56, 0.67, 1, 2, 3],
            [1, 3, 1, 0, 6, 1, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            [1, 3, 1, 0, 6, 1, 1, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            [0, 3, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            [0, 3, 0, 0, 6, 0, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            [1, 3, 1, 0, 6, 1, 2, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            [0, 3, 1, 0, 6, 0, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            [2, 1, 0, 4, 0, 4, 3, 105, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0, 0, 0],
            [2, 1, 0, 4, 0, 4, 3, 105, 0.76, 0.75, 0.74, 0.73, 0.72, 0.71, 0, 0, 0],
            [3, 1, 0, 4, 0, 4, 3, 105, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0, 4, 2],
            [3, 1, 0, 4, 0, 4, 3, 105, 0.76, 0.75, 0.74, 0.73, 0.72, 0.71, 1, 3, 0],
            [0, 0, 4, 3, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0],
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
            [0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            [0, 0, 4, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            [1, 3, 1, 0, 6, 0, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            [2, 1, 0, 4, 0, 4, 3, 105, 0.12, 0.23, 0.34, 0.45, 0.56, 0.67, 0, 0, 0],
            [3, 1, 0, 4, 0, 4, 3, 105, 0.12, 0.23, 0.34, 0.45, 0.56, 0.67, 1, 2, 3],
            [1, 3, 1, 0, 6, 1, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            [1, 3, 1, 0, 6, 1, 1, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            [0, 3, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            [0, 3, 0, 0, 6, 0, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            [1, 3, 1, 0, 6, 1, 2, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            [0, 3, 1, 0, 6, 0, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            [2, 1, 0, 4, 0, 4, 3, 105, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0, 0, 0],
            [2, 1, 0, 4, 0, 4, 3, 105, 0.76, 0.75, 0.74, 0.73, 0.72, 0.71, 0, 0, 0],
            [3, 1, 0, 4, 0, 4, 3, 105, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0, 4, 2],
            [3, 1, 0, 4, 0, 4, 3, 105, 0.76, 0.75, 0.74, 0.73, 0.72, 0.71, 1, 3, 0],
            [0, 0, 4, 3, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0],
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
            f"LatticeParameters = {readables[2]};"
            f"MillerIndices = {readables[3]}"
        )
    # Get policy states from env.states2policy
    states_readable = [env.state2readable(state) for state in states]
    for readable, readable_expected in zip(states_readable, states_readable_expected):
        assert readable == readable_expected
