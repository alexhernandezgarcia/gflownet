import pytest
import torch
import numpy as np
from gflownet.envs.grid import Grid
import common

@pytest.fixture
def env():
        return Grid(n_dim=3, length=5)

def test__get_parents_step_get_mask__are_compatible(env):
    return common.test__get_parents_step_get_mask__are_compatible(env)


