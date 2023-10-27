import common
import pytest
import torch

from gflownet.envs.seqs.sequence import Sequence


@pytest.fixture
def env():
    return Sequence(tokens=[0, 1], max_length=5, device="cpu")


@pytest.fixture
def env_default():
    return Sequence()


@pytest.mark.parametrize(
    "tokens",
    [
        [0, 1],
        [-2, -1, 0, 1, 2],
        (0, 1),
        (-2, -1, 0, 1, 2),
        [0, 1, 1],
    ],
)
def test__environment_initializes_properly(tokens):
    env = Sequence(tokens=tokens, device="device")
    assert True


@pytest.mark.parametrize(
    "action_space",
    [
        [(0,), (1,), (-1,)],
    ],
)
def test__get_action_space__returns_expected(env, action_space):
    assert set(action_space) == set(env.action_space)
