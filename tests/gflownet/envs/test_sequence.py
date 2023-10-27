import common
import pytest
import torch

from gflownet.envs.seqs.sequence import Sequence
from gflownet.utils.common import tlong


@pytest.fixture
def env():
    return Sequence(tokens=[-2, -1, 0, 1, 2], max_length=5, device="cpu")


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
        [(0,), (1,), (2,), (3,), (4,), (-1,)],
    ],
)
def test__get_action_space__returns_expected(env, action_space):
    assert set(action_space) == set(env.action_space)


@pytest.mark.parametrize(
    "state, parents_expected, parents_a_expected",
    [
        (
            [-2, -2, -2, -2, -2],
            [],
            [],
        ),
        (
            [0, -2, -2, -2, -2],
            [[-2, -2, -2, -2, -2]],
            [(0,)],
        ),
        (
            [0, 2, 1, 3, -2],
            [[0, 2, 1, -2, -2]],
            [(3,)],
        ),
    ],
)
def test__get_parents__returns_expected(
    env, state, parents_expected, parents_a_expected
):
    state = tlong(state, device=env.device)
    parents_expected = [tlong(parent, device=env.device) for parent in parents_expected]
    parents, parents_a = env.get_parents(state)
    for p, p_e in zip(parents, parents_expected):
        assert torch.equal(p, p_e)
    for p_a, p_a_e in zip(parents_a, parents_a_expected):
        assert p_a == p_a_e


@pytest.mark.parametrize(
    "state, action, next_state",
    [
        (
            [-2, -2, -2, -2, -2],
            (0,),
            [0, -2, -2, -2, -2],
        ),
        (
            [0, -2, -2, -2, -2],
            (2,),
            [0, 2, -2, -2, -2],
        ),
        (
            [0, 2, -2, -2, -2],
            (1,),
            [0, 2, 1, -2, -2],
        ),
        (
            [0, 2, 1, -2, -2],
            (0,),
            [0, 2, 1, 0, -2],
        ),
        (
            [0, 2, 1, 0, -2],
            (3,),
            [0, 2, 1, 0, 3],
        ),
        (
            [0, 2, 1, 0, 3],
            (-1,),
            [0, 2, 1, 0, 3],
        ),
        (
            [0, 2, 1, -2, -2],
            (-1,),
            [0, 2, 1, -2, -2],
        ),
    ],
)
def test__step__returns_expected(env, state, action, next_state):
    env.set_state(tlong(state, device=env.device))
    env.step(action)
    assert torch.equal(env.state, tlong(next_state, device=env.device))
