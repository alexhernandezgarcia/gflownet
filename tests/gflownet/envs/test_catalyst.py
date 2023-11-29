import warnings

import common
import numpy as np
import pytest
import torch
from torch import Tensor

from gflownet.envs.crystals.catalyst import Catalyst, Stage
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
